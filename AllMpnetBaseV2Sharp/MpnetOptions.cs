namespace AllMpnetBaseV2Sharp;

/// <summary>
/// Configuration options for AllMpnetBaseV2Embedder.
/// </summary>
public sealed class MpnetOptions
{
    /// <summary>
    /// Maximum number of tokens per input sequence.
    /// Defaults to 384 (SentenceTransformers default).
    /// </summary>
    public int MaxTokens { get; init; } = 384;

    /// <summary>
    /// Whether to L2-normalize output embeddings.
    /// Enabled by default.
    /// </summary>
    public bool Normalize { get; init; } = true;

    /// <summary>
    /// Optional custom ONNX model path.
    /// </summary>
    public string? ModelPath { get; init; }

    /// <summary>
    /// Optional custom tokenizer.json path.
    /// </summary>
    public string? TokenizerPath { get; init; }
}
