using System;
using System.Linq;
using Xunit;
using AllMpnetBaseV2Sharp;

namespace AllMpnetBaseV2Sharp.Tests;
public class EmbeddingTests(EmbedderFixture fixture) : IClassFixture<EmbedderFixture>
{
    private readonly AllMpnetBaseV2Embedder _embedder = fixture.Embedder;

    [Fact]
    public void Encode_ReturnsVectorOfExpectedSize()
    {
        var embedding = _embedder.Encode("Hello world");

        Assert.NotNull(embedding);
        Assert.Equal(768, embedding.Length);
    }

    [Theory]
    [InlineData("The quick brown fox")]
    [InlineData("Hello world")]
    [InlineData("A cat sits on the mat")]
    public void Encode_IsDeterministic(string text)
    {
        var a = _embedder.Encode(text);
        var b = _embedder.Encode(text);

        Assert.Equal(a.Length, b.Length);

        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i], precision: 6);
    }

    [Fact]
    public void Encode_EmbeddingsAreNormalized()
    {
        var embedding = _embedder.Encode("Test sentence");

        double norm = Math.Sqrt(embedding.Sum(x => x * x));
        Assert.InRange(norm, 0.9999, 1.0001);
    }

    [Fact]
    public void SimilarSentences_HaveHigherSimilarity()
    {
        var v1 = _embedder.Encode("A cat sits on the mat");
        var v2 = _embedder.Encode("A kitten is sitting on a rug");
        var v3 = _embedder.Encode("The stock market crashed yesterday");

        double sim12 = CosineSimilarity(v1, v2);
        double sim13 = CosineSimilarity(v1, v3);

        Assert.True(sim12 > sim13);
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length.");

        double dot = 0, na = 0, nb = 0;

        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }

        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }
}