from sie_server.adapters.sglang.generation import SGLangGenerationAdapter


class SGLangGemmaAdapter(SGLangGenerationAdapter):
    """Bundle-routing seam for the Gemma worker image.

    Behaviour is identical to :class:`SGLangGenerationAdapter` — Gemma 4 is
    served through the same ``sglang serve`` subprocess with the ``gemma4``
    reasoning/tool parsers set via the model YAML. The only reason this
    subclass exists is routing: bundle compatibility keys on the adapter
    *module* path, and the ``sglang`` bundle declares only
    ``sie_server.adapters.sglang.generation``. Giving Gemma 4 a distinct
    module lets the ``gemma`` bundle (sglang 0.5.13 + transformers 5.8) own
    it without making every Qwen3.x model compatible with that bundle too —
    so the Qwen serving stack on the ``sglang`` bundle stays untouched.
    """
