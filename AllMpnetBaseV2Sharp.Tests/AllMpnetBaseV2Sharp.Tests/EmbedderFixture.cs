using System;
using System.Linq;
using Xunit;
using AllMpnetBaseV2Sharp;

namespace AllMpnetBaseV2Sharp.Tests;

public class EmbedderFixture : IDisposable
{
    public AllMpnetBaseV2Embedder Embedder { get; }

    public EmbedderFixture()
    {
        var options = new MpnetOptions
        {
            ModelPath = "model.onnx",
            TokenizerPath = "tokenizer.json"
        };

        Embedder = new AllMpnetBaseV2Embedder(options);
    }

    public void Dispose()
    {
        Embedder.Dispose();
    }
}