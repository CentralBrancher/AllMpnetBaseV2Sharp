using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.HuggingFace.Tokenizer;
using Tokenizers.HuggingFace.Errors;
using AllMpnetBaseV2Sharp.Utilities;

namespace AllMpnetBaseV2Sharp;

/// <summary>
/// Sentence embedding generator for sentence-transformers/all-mpnet-base-v2.
/// </summary>
public sealed class AllMpnetBaseV2Embedder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly Tokenizer _tokenizer;
    private readonly MpnetOptions _options;

    public int MaxTokens => _options.MaxTokens;

    public AllMpnetBaseV2Embedder(MpnetOptions? options = null)
    {
        _options = options ?? new MpnetOptions();

        string basePath = AppContext.BaseDirectory;
        string modelPath = _options.ModelPath
            ?? Path.Combine(basePath, "model", "model.onnx");
        string tokenizerPath = _options.TokenizerPath
            ?? Path.Combine(basePath, "model", "tokenizer.json");

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}");

        if (!File.Exists(tokenizerPath))
            throw new FileNotFoundException($"Tokenizer not found: {tokenizerPath}");

        _session = new InferenceSession(modelPath);
        _tokenizer = Tokenizer.FromFile(tokenizerPath);
    }

    /// <summary>
    /// Encode a single sentence into a normalized embedding.
    /// </summary>
    public float[] Encode(string text)
        => Encode(new[] { text })[0];

    /// <summary>
    /// Encode multiple sentences into normalized embeddings.
    /// </summary>
    public float[][] Encode(IEnumerable<string> texts)
    {
        var inputs = texts as string[] ?? texts.ToArray();
        if (inputs.Length == 0)
            return Array.Empty<float[]>();

        var encodings = inputs.Select(SafeEncode).ToList();
        int seqLen = Math.Min(MaxTokens, encodings.Max(e => e.Ids.Count));
        int batch = encodings.Count;

        var inputIds = new DenseTensor<long>(new[] { batch, seqLen });
        var attentionMask = new DenseTensor<long>(new[] { batch, seqLen });

        for (int i = 0; i < batch; i++)
        {
            var ids = encodings[i].Ids;
            var mask = encodings[i].AttentionMask;

            for (int j = 0; j < seqLen; j++)
            {
                if (j < ids.Count)
                {
                    inputIds[i, j] = ids[j];
                    attentionMask[i, j] = j < mask.Count ? mask[j] : 1;
                }
                else
                {
                    inputIds[i, j] = 1; // [PAD]
                    attentionMask[i, j] = 0;
                }
            }
        }

        var inputsOnnx = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };

        if (_session.InputMetadata.ContainsKey("token_type_ids"))
        {
            inputsOnnx.Add(
                NamedOnnxValue.CreateFromTensor(
                    "token_type_ids",
                    new DenseTensor<long>(new[] { batch, seqLen })
                )
            );
        }

        using var results = _session.Run(inputsOnnx);
        var tensor = results[0].AsTensor<float>();
        var dims = tensor.Dimensions.ToArray();

        var embeddings = new float[batch][];

        if (dims.Length == 2)
        {
            int dim = dims[1];
            var flat = tensor.ToArray();

            for (int i = 0; i < batch; i++)
            {
                var vec = new float[dim];
                Array.Copy(flat, i * dim, vec, 0, dim);
                if (_options.Normalize)
                    VectorMath.L2NormalizeInPlace(vec);
                embeddings[i] = vec;
            }
        }
        else
        {
            int dim = dims[2];

            for (int i = 0; i < batch; i++)
            {
                var vec = new float[dim];
                int count = 0;

                for (int j = 0; j < seqLen; j++)
                {
                    if (attentionMask[i, j] == 0)
                        continue;

                    for (int k = 0; k < dim; k++)
                        vec[k] += tensor[i, j, k];

                    count++;
                }

                if (count > 0)
                    for (int k = 0; k < dim; k++)
                        vec[k] /= count;

                if (_options.Normalize)
                    VectorMath.L2NormalizeInPlace(vec);

                embeddings[i] = vec;
            }
        }

        return embeddings;
    }

    private Encoding SafeEncode(string text)
    {
        try
        {
            return _tokenizer.Encode(
                input: text,
                addSpecialTokens: true,
                includeAttentionMask: true
            ).First();
        }
        catch (TokenizerEncodingException ex)
        {
            throw new InvalidOperationException("Tokenization failed.", ex);
        }
    }

    public void Dispose()
    {
        _session.Dispose();
        _tokenizer.Dispose();
    }
}
