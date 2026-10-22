"""Tests for the turn-amnesia fix: image rejection detection + in-place strip."""
from providers.openai_compatible import OpenAICompatibleProvider as P

def run():
    ok = 0
    rej = P._is_image_rejection
    for msg in ["No endpoints found that support image input",
                "Error 400: model does not support image input",
                "image_url is not supported for this model"]:
        assert rej(Exception(msg)), msg; ok += 1
    for msg in ["rate limit exceeded", "401 Unauthorized", "context length exceeded"]:
        assert not rej(Exception(msg)), msg; ok += 1
    fake = type("F", (), {"model": "m"})()
    img = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]
    hist = [{"role": "user", "content": list(img)},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "plain"}]
    wire = [{"role": "user", "content": list(img)}]
    n = P._strip_images(fake, hist, wire, None)
    assert n == 2; ok += 1
    assert "image omitted" in hist[0]["content"] and "image omitted" in wire[0]["content"]; ok += 1
    assert hist[1]["content"] == "reply" and hist[2]["content"] == "plain"; ok += 1
    assert P._strip_images(fake, hist, wire) == 0; ok += 1  # idempotent
    print(f"{ok}/{ok} passed"); return ok

if __name__ == "__main__":
    run()
