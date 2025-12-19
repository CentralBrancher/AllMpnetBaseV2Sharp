namespace AllMpnetBaseV2Sharp.Utilities;

internal static class VectorMath
{
    public static void L2NormalizeInPlace(float[] v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++)
            sum += (double)v[i] * v[i];

        if (sum <= 0)
            return;

        double norm = Math.Sqrt(sum);
        for (int i = 0; i < v.Length; i++)
            v[i] = (float)(v[i] / norm);
    }
}
