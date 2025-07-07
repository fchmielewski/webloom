from loom.model import LoomModel


def test_encode_prompt_truncates():
    model = LoomModel('sshleifer/tiny-gpt2', device='cpu')
    long_prompt = ' '.join(['word'] * (model.tokenizer.model_max_length + 10))
    tokens = model._encode_prompt(long_prompt, room_for=5)
    assert tokens.shape[1] <= model.tokenizer.model_max_length - 5


def test_generate_returns_strings():
    model = LoomModel('sshleifer/tiny-gpt2', device='cpu')
    res = model.generate('Hello', max_new_tokens=1, num_return_sequences=2)
    assert len(res) == 2
    assert all(isinstance(r, str) for r in res)


def test_unavailable_env_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv('LOOM_DEVICE', 'cuda')
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    model = LoomModel('sshleifer/tiny-gpt2')
    assert model.device == 'cpu'