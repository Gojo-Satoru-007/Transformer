import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scratch_gpt.tokenizer import BPETokenizer, EOT_TOKEN

CORPUS = ("the quick brown fox jumps over the lazy dog. " * 50 + "naïve café 123, don't! ") * 5


def test_roundtrip_and_vocab():
    tok = BPETokenizer().train(CORPUS, vocab_size=300, verbose=False)
    assert tok.vocab_size == 300 and tok.eot_id == 299
    for s in ["the quick brown fox", "naïve café 123", "don't  stop!\n", "unseen ZZZ 😀"]:
        assert tok.decode(tok.encode(s)) == s


def test_compresses_and_eot():
    tok = BPETokenizer().train(CORPUS, vocab_size=300, verbose=False)
    s = "the quick brown fox jumps over the lazy dog."
    assert len(tok.encode(s)) < len(s.encode())
    ids = tok.encode("a" + EOT_TOKEN + "b")
    assert ids.count(tok.eot_id) == 1


def test_save_load(tmp_path):
    tok = BPETokenizer().train(CORPUS, vocab_size=300, verbose=False)
    tok.save(tmp_path / "t.json")
    tok2 = BPETokenizer.load(tmp_path / "t.json")
    assert tok2.encode("the lazy dog") == tok.encode("the lazy dog")
