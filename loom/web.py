"""Flask web interface for Loom GPT-2."""

from flask import Flask, render_template, request, jsonify, Response
from flask import redirect, url_for
from markupsafe import Markup
import json
import threading
import gc
import torch
from .model import LoomModel
from .database import Database


class LoomApp:
    """Flask application encapsulating model and database."""

    def __init__(self) -> None:
        self.app = Flask(__name__)
        self.db = Database()
        self.models: dict[str, LoomModel] = {}
        self.model_name = "sshleifer/tiny-gpt2"
        self.model = self._load_model(self.model_name)
        self.loading = False
        self._add_routes()

    def _load_model(self, name: str) -> LoomModel:
        if name not in self.models:
            # Remove previously loaded models to free memory and avoid crashes
            for other in list(self.models.keys()):
                if other != name:
                    del self.models[other]
            gc.collect()
            torch.cuda.empty_cache()
            self.models[name] = LoomModel(name)
        return self.models[name]

    def _add_routes(self) -> None:
        @self.app.route('/', methods=['GET'])
        def index():
            tree_id = request.args.get('tree_id')
            selected_id = request.args.get('node_id', default=0, type=int)
            tree = None
            length = None
            temperature = None
            model_name = None
            variants = None
            if tree_id:
                rec = self.db.get_tree(int(tree_id))
                if rec:
                    try:
                        tree = json.loads(rec.data)
                    except json.JSONDecodeError:
                        tree = None
                    length = rec.length
                    temperature = rec.temperature
                    model_name = rec.model
                    variants = rec.variants
            else:
                blank = self.db.get_blank_tree()
                if blank:
                    return redirect(url_for('index', tree_id=blank.id))
                empty = json.dumps([{"id": 0, "text": "", "parent": None}])
                tree_id = self.db.add_tree(empty, model=self.model_name)
                tree = json.loads(empty)
                length = 100
                temperature = 0.8
                model_name = self.model_name
                variants = 3
            return render_template(
                "index.html",
                tree=tree,
                tree_id=tree_id,
                selected_id=selected_id,
                length=length,
                temperature=temperature,
                model_name=model_name,
                variants=variants,
            )

        @self.app.route('/generate', methods=['POST'])
        def generate():
            if self.loading:
                return jsonify({'error': 'model loading'}), 503
            data = request.get_json(force=True)
            prompt = data.get('prompt', '')
            num = int(data.get('num_return_sequences', 3))
            max_new_tokens = int(
                data.get('max_new_tokens', data.get('max_length', 100))
            )
            temperature = float(data.get('temperature', 0.8))
            responses = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num,
                temperature=temperature,
            )
            continuations = [
                r[len(prompt):] if r.startswith(prompt) else r
                for r in responses
            ]
            return jsonify({'continuations': continuations})

        @self.app.route('/generate_stream')
        def generate_stream():
            if self.loading:
                return Response("model loading", status=503)
            prompt = request.args.get('prompt', '')
            max_new_tokens = int(
                request.args.get('max_new_tokens', request.args.get('max_length', 100))
            )
            temperature = float(request.args.get('temperature', 0.8))

            def event_stream():
                for token in self.model.stream(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ):
                    yield "data: " + json.dumps({'token': token}) + "\n\n"
                yield "data: [DONE]\n\n"

            return Response(event_stream(), mimetype='text/event-stream')

        @self.app.route('/save', methods=['POST'])
        def save():
            data = request.get_json(force=True)
            tree = data.get('tree')
            tree_id = data.get('tree_id')
            length = data.get('length')
            temperature = data.get('temperature')
            model_name = data.get('model')
            variants = data.get('variants')
            if tree is None:
                return jsonify({'status': 'error', 'message': 'no tree'}), 400
            serialized = json.dumps(tree)
            if tree_id:
                self.db.update_tree(
                    int(tree_id),
                    serialized,
                    length=int(length) if length is not None else None,
                    temperature=float(temperature) if temperature is not None else None,
                    model=model_name,
                    variants=int(variants) if variants is not None else None,
                )
            else:
                tree_id = self.db.add_tree(
                    serialized,
                    length=int(length) if length is not None else 100,
                    temperature=float(temperature) if temperature is not None else 0.8,
                    model=model_name or "sshleifer/tiny-gpt2",
                    variants=int(variants) if variants is not None else 3,
                )
            return jsonify({'status': 'ok', 'id': tree_id})

        @self.app.route('/new_tree', methods=['POST'])
        def new_tree():
            blank = self.db.get_blank_tree()
            if blank:
                return jsonify({'id': blank.id})
            empty = json.dumps([{'id': 0, 'text': '', 'parent': None}])
            tree_id = self.db.add_tree(empty, model=self.model_name)
            return jsonify({'id': tree_id})

        @self.app.route('/set_model', methods=['POST'])
        def set_model():
            data = request.get_json(force=True)
            model_name = data.get('model', 'gpt2')
            if model_name == self.model_name:
                return jsonify({'status': 'ready'})
            if model_name in self.models:
                self.model = self.models[model_name]
                self.model_name = model_name
                return jsonify({'status': 'ready'})

            def loader() -> None:
                self.model = self._load_model(model_name)
                self.model_name = model_name
                self.loading = False

            self.loading = True
            threading.Thread(target=loader, daemon=True).start()
            return jsonify({'status': 'loading'})

        @self.app.route('/model_status')
        def model_status():
            return jsonify({'loading': self.loading})

        @self.app.route('/delete_tree/<int:tree_id>', methods=['POST'])
        def delete_tree(tree_id: int):
            self.db.delete_tree(tree_id)
            return jsonify({'status': 'ok'})

        @self.app.route('/duplicate_tree/<int:tree_id>', methods=['POST'])
        def duplicate_tree(tree_id: int):
            new_id = self.db.duplicate_tree(tree_id)
            return jsonify({'status': 'ok', 'id': new_id})

        @self.app.route('/history')
        def history():
            current_id = request.args.get("current_id")
            node_id = request.args.get("node_id")
            raw = self.db.get_all_trees()
            records = []
            for r in raw:
                preview = ""
                try:
                    data = json.loads(r.data)
                    longest = max((n.get('text', '') for n in data), key=len, default="")
                    preview = longest.replace("\n", " ")
                except Exception:
                    pass
                records.append({
                    'id': r.id,
                    'timestamp': r.timestamp.strftime('%Y-%m-%d') if r.timestamp else '',
                    'preview': preview,
                })
            return render_template(
                'history.html',
                records=records,
                current_id=current_id,
                node_id=node_id,
            )

        @self.app.route('/tree')
        def tree_view():
            tree_id = request.args.get('tree_id')
            node_id = request.args.get('node_id')
            tree_data = None
            if tree_id:
                rec = self.db.get_tree(int(tree_id))
                if rec:
                    try:
                        tree_data = json.loads(rec.data)
                    except Exception:
                        tree_data = None
            if tree_data is None:
                blank = self.db.get_blank_tree()
                if blank:
                    return redirect(url_for('tree_view', tree_id=blank.id))
                empty = json.dumps([{"id": 0, "text": "", "parent": None}])
                tree_id = self.db.add_tree(empty, model=self.model_name)
                tree_data = json.loads(empty)

            id_map = {n['id']: n for n in tree_data}
            children = {}
            for n in tree_data:
                children.setdefault(n['parent'], []).append(n)

            def build_html(node_id: int) -> str:
                node = id_map[node_id]
                if node['parent'] is None:
                    text = '[start]'
                else:
                    parent_text = id_map[node['parent']]['text']
                    text = node['text'][len(parent_text):]
                url = url_for('index', tree_id=tree_id, node_id=node_id)
                html = f'<li><a href="{url}">{text}</a>'
                if node_id in children:
                    html += '<ul>'
                    for child in children[node_id]:
                        html += build_html(child['id'])
                    html += '</ul>'
                html += '</li>'
                return html

            tree_html = Markup('<ul class="tree">' + build_html(0) + '</ul>')
            return render_template(
                'tree.html', tree_html=tree_html, tree_id=tree_id, node_id=node_id
            )

    def run(self, **kwargs) -> None:
        """Run the Flask development server."""
        self.app.run(**kwargs)
