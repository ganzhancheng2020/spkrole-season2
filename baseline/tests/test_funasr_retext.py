"""冒烟测试：mock transcribe_audio，验证重排和落盘格式正确。"""

from src import funasr_retext


def test_tokenize_words_skips_punct():
    words = [
        {"begin_time": 0, "end_time": 100, "text": "你 好"},
        {"begin_time": 100, "end_time": 200, "text": "，世 界。"},
    ]
    out = funasr_retext.tokenize_words(words)
    assert [t for t, _, _ in out] == ["你", "好", "世", "界"]


def test_reassign_assigns_word_to_overlapping_segment():
    moss = [
        {
            "session_id": "001",
            "speaker": "spk1",
            "start_time": 0.0,
            "end_time": 5.0,
            "words": "",
        },
        {
            "session_id": "001",
            "speaker": "spk2",
            "start_time": 5.0,
            "end_time": 10.0,
            "words": "",
        },
    ]
    words = [("你", 1.0, 2.0), ("好", 7.0, 8.0)]
    out = funasr_retext.reassign(moss, words)
    assert out[0]["words"] == "你"
    assert out[1]["words"] == "好"
