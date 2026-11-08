// The exact GELU, x (1 + erf(x / sqrt 2)) / 2, with erf as the rational fit
// the CPU uses (native_engine::math).
float gelu(float x) {
    float z = clamp(x * 0.70710678, -4.0, 4.0);
    float z2 = z * z;
    float p = -2.7261423e-10;
    p = p * z2 + 2.7706814e-8;
    p = p * z2 - 2.101024e-6;
    p = p * z2 - 5.6925064e-5;
    p = p * z2 - 7.3499063e-4;
    p = p * z2 - 2.9546e-3;
    p = p * z2 - 1.6096033e-2;
    float q = -1.4566072e-5;
    q = q * z2 - 2.1337406e-4;
    q = q * z2 - 1.682827e-3;
    q = q * z2 - 7.373329e-3;
    q = q * z2 - 1.426474e-2;
    return 0.5 * x * (1.0 + z * p / q);
}
