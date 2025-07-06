import json
from loom.database import Database


def test_add_and_get_record():
    db = Database("sqlite:///:memory:")
    db.add_record("prompt", "response")
    records = db.get_all()
    assert len(records) == 1
    assert records[0].prompt == "prompt"
    assert records[0].response == "response"


def test_tree_operations():
    db = Database("sqlite:///:memory:")
    tree_json = json.dumps([{"id": 0, "text": "", "parent": None}])
    tid = db.add_tree(tree_json, length=5, temperature=0.5, model="model", variants=2)
    rec = db.get_tree(tid)
    assert rec.length == 5
    assert rec.temperature == 0.5
    assert rec.model == "model"
    assert rec.variants == 2

    dup_id = db.duplicate_tree(tid)
    dup = db.get_tree(dup_id)
    assert dup.data == rec.data
    assert dup.length == rec.length

    db.delete_tree(tid)
    assert db.get_tree(tid) is None
