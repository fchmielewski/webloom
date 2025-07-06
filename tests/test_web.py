import json
from unittest.mock import patch
from loom.web import LoomApp
from loom.database import Database


class DummyModel:
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, max_new_tokens: int = 100, num_return_sequences: int = 1, temperature: float = 0.8):
        return [prompt + '_gen'] * num_return_sequences

    def stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8):
        for c in 'x':
            yield c


class InMemoryDB(Database):
    def __init__(self):
        super().__init__('sqlite:///:memory:')


def test_index_creates_tree():
    with patch('loom.web.LoomModel', DummyModel), patch('loom.web.Database', InMemoryDB):
        app = LoomApp()
        client = app.app.test_client()
        response = client.get('/')
        assert response.status_code == 200
        assert b'<html' in response.data


def test_generate_endpoint():
    with patch('loom.web.LoomModel', DummyModel), patch('loom.web.Database', InMemoryDB):
        app = LoomApp()
        client = app.app.test_client()
        res = client.post('/generate', json={'prompt': 'hi', 'num_return_sequences': 2})
        assert res.status_code == 200
        data = res.get_json()
        assert data['continuations'] == ['_gen', '_gen']
