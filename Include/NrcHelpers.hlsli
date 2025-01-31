/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __NRC_HELPERS_HLSL__
#define __NRC_HELPERS_HLSL__

#include "NrcStructures.h"

#ifndef __cplusplus

/**
 *   This file contains utility functions used by NRC shader code, mostly related to packing/unpacking
 *   of NRC data structures. It shouldn't be necessary to modify this code for integrating NRC.
 */

/** Convert float value to 16-bit unorm (unsafe version).
    \param[in] v Value assumed to be in [0,1].
    \return 16-bit unorm in low bits, high bits all zeros.
*/
uint NrcPackUnorm16Unsafe(float v)
{
    return (uint)trunc(v * 65535.f + 0.5f);
}

/** Convert float value to 16-bit unorm.
    Values outside [0,1] are clamped and NaN is encoded as zero.
    \return 16-bit unorm in low bits, high bits all zeros.
*/
uint NrcPackUnorm16(float v)
{
    v = isnan(v) ? 0.f : saturate(v);
    return NrcPackUnorm16Unsafe(v);
}

/** Pack two floats into 16-bit unorm values in a dword.
 */
uint NrcPackUnorm2x16(float2 v)
{
    return (NrcPackUnorm16(v.y) << 16) | NrcPackUnorm16(v.x);
}

/** Unpack two 16-bit unorm values from a dword.
*   \param[in] packed Two 16-bit unorm in low/high bits.
    \return Two float values in [0,1].
*/
float2 NrcUnpackUnorm2x16(uint packed)
{
    return float2(packed & 0xffff, packed >> 16) * (1.f / 65535);
}

/** Convert float value to 16-bit snorm value.
    Values outside [-1,1] are clamped and NaN is encoded as zero.
    \return 16-bit snorm value in low bits, high bits are all zeros or ones depending on sign.
*/
int NrcFloatToSnorm16(float v)
{
    v = isnan(v) ? 0.f : min(max(v, -1.f), 1.f);
    return (int)trunc(v * 32767.f + (v >= 0.f ? 0.5f : -0.5f));
}

/** Pack two floats into 16-bit snorm values in the lo/hi bits of a dword.
    \return Two 16-bit snorm in low/high bits.
*/
uint NrcPackSnorm2x16(precise float2 v)
{
    return (NrcFloatToSnorm16(v.x) & 0x0000ffff) | (NrcFloatToSnorm16(v.y) << 16);
}

/** Unpack two 16-bit snorm values from the lo/hi bits of a dword.
    \param[in] packed Two 16-bit snorm in low/high bits.
    \return Two float values in [-1,1].
*/
float2 NrcUnpackSnorm2x16(uint packed)
{
    int2 bits = int2(packed << 16, packed) >> 16;
    precise float2 unpacked = max((float2)bits / 32767.f, -1.0f);
    return unpacked;
}

/** Helper function to reflect the folds of the lower hemisphere
    over the diagonals in the octahedral map.
*/
inline float2 NrcOctWrap(float2 v)
{
    return float2((1.f - abs(v.y)) * (v.x >= 0.f ? 1.f : -1.f), (1.f - abs(v.x)) * (v.y >= 0.f ? 1.f : -1.f));
}

/** Converts point in the octahedral map to normalized direction (non-equal area, signed normalized).
    \param[in] p Position in octahedral map in [-1,1] for each component.
    \return Normalized direction.
*/
inline float3 NrcOctToDirection(float2 p)
{
    float3 n = float3(p.x, p.y, 1.f - abs(p.x) - abs(p.y));
    float2 tmp = (n.z < 0.0) ? NrcOctWrap(float2(n.x, n.y)) : float2(n.x, n.y);
    n.x = tmp.x;
    n.y = tmp.y;
    return normalize(n);
}

/** Converts normalized direction to the octahedral map (non-equal area, signed normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [-1,1] for each component.
*/
float2 NrcDirectionToOct(float3 n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane.
    float2 p = n.xy * (1.f / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.f) ? NrcOctWrap(p) : p;
    return p;
}

/** Encode a normal packed as 2x 16-bit snorms in the octahedral mapping.
 */
uint NrcEncodeNormal2x16(float3 normal)
{
    float2 octNormal = NrcDirectionToOct(normal);
    return NrcPackSnorm2x16(octNormal);
}

/** Decode a normal packed as 2x 16-bit snorms in the octahedral mapping.
 */
float3 NrcDecodeNormal2x16(uint packedNormal)
{
    float2 octNormal = NrcUnpackSnorm2x16(packedNormal);
    return NrcOctToDirection(octNormal);
}

/** Unpack three positive floats from a dword.
    https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_R11G11B10.hlsli
*/
float3 NrcUnpackR11G11B10(uint packed)
{
    float r = f16tof32((packed << 4) & 0x7FF0);
    float g = f16tof32((packed >> 7) & 0x7FF0);
    float b = f16tof32((packed >> 17) & 0x7FE0);
    return float3(r, g, b);
}

/** Pack three positive floats into a dword.
    https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_R11G11B10.hlsli
*/
uint NrcPackR11G11B10(float3 v)
{
    // Clamp upper bound so that it doesn't accidentally round up to INF
    v = min(v, asfloat(0x477C0000));
    // Exponent=15, Mantissa=1.11111
    uint r = ((f32tof16(v.x) + 8) >> 4) & 0x000007ff;
    uint g = ((f32tof16(v.y) + 8) << 7) & 0x003ff800;
    uint b = ((f32tof16(v.z) + 16) << 17) & 0xffc00000;
    return r | g | b;
}

/** Transforms an RGB color in Rec.709 to CIE XYZ.
 */
float3 NrcRgbToXyzRec709(float3 c)
{
    // clang-format off
    static const float3x3 M =
    {
        0.4123907992659595, 0.3575843393838780, 0.1804807884018343,
        0.2126390058715104, 0.7151686787677559, 0.0721923153607337,
        0.0193308187155918, 0.1191947797946259, 0.9505321522496608
    };
    // clang-format on
    return mul(M, c);
}

/** Transforms an XYZ color to RGB in Rec.709.
 */
float3 NrcXyzToRgbRec709(float3 c)
{
    // clang-format off
    static const float3x3 M =
    {
        3.240969941904522, -1.537383177570094, -0.4986107602930032,
        -0.9692436362808803, 1.875967501507721, 0.04155505740717569,
        0.05563007969699373, -0.2039769588889765, 1.056971514242878
    };
    // clang-format on
    return mul(M, c);
}

/** Encode an RGB color into a 32-bit LogLuv HDR format.
    The supported luminance range is roughly 10^-6..10^6 in 0.17% steps.

    The log-luminance is encoded with 14 bits and chroma with 9 bits each.
    This was empirically more accurate than using 8 bit chroma.
    Black (all zeros) is handled exactly.
*/
uint NrcEncodeLogLuvHdr(float3 color)
{
    // Convert RGB to XYZ.
    float3 XYZ = NrcRgbToXyzRec709(color);

    // Encode log2(Y) over the range [-20,20) in 14 bits (no sign bit).
    // TODO: Fast path that uses the bits from the fp32 representation directly.
    float logY = 409.6f * (log2(XYZ.y) + 20.f); // -inf if Y==0
    uint Le = (uint)clamp(logY, 0.f, 16383.f);

    // Early out if zero luminance to avoid NaN in chroma computation.
    // Note Le==0 if Y < 9.55e-7. We'll decode that as exactly zero.
    if (Le == 0)
    {
        return 0;
    }

    // Compute chroma (u,v) values by:
    //  x = X / (X + Y + Z)
    //  y = Y / (X + Y + Z)
    //  u = 4x / (-2x + 12y + 3)
    //  v = 9y / (-2x + 12y + 3)
    //
    // These expressions can be refactored to avoid a division by:
    //  u = 4X / (-2X + 12Y + 3(X + Y + Z))
    //  v = 9Y / (-2X + 12Y + 3(X + Y + Z))
    //
    float invDenom = 1.f / (-2.f * XYZ.x + 12.f * XYZ.y + 3.f * (XYZ.x + XYZ.y + XYZ.z));
    float2 uv = float2(4.f, 9.f) * XYZ.xy * invDenom;

    // Encode chroma (u,v) in 9 bits each.
    // The gamut of perceivable uv values is roughly [0,0.62], so scale by 820 to get 9-bit values.
    uint2 uve = (uint2)clamp(820.f * uv, 0.f, 511.f);

    return (Le << 18) | (uve.x << 9) | uve.y;
}

/** Decode an RGB color stored in a 32-bit LogLuv HDR format.
    See encodeLogLuvHDR() for details.
*/
float3 NrcDecodeLogLuvHdr(uint packedColor)
{
    // Decode luminance Y from encoded log-luminance.
    uint Le = packedColor >> 18;
    if (Le == 0)
    {
        return float3(0.f, 0.f, 0.f);
    }

    float logY = (float(Le) + 0.5f) / 409.6f - 20.f;
    float Y = pow(2.f, logY);

    // Decode normalized chromaticity xy from encoded chroma (u,v).
    //
    //  x = 9u / (6u - 16v + 12)
    //  y = 4v / (6u - 16v + 12)
    //
    uint2 uve = uint2(packedColor >> 9, packedColor) & 0x1ff;
    float2 uv = (float2(uve) + 0.5f) / 820.f;

    float invDenom = 1.f / (6.f * uv.x - 16.f * uv.y + 12.f);
    float2 xy = float2(9.f, 4.f) * uv * invDenom;

    // Convert chromaticity to XYZ and back to RGB.
    //  X = Y / y * x
    //  Z = Y / y * (1 - x - y)
    //
    float s = Y / xy.y;
    float3 XYZ = { s * xy.x, Y, s * (1.f - xy.x - xy.y) };

    // Convert back to RGB and clamp to avoid out-of-gamut colors.
    return max(NrcXyzToRgbRec709(XYZ), 0.f);
}

/** Converts Cartesian coordinates to spherical coordinates (unsigned normalized).
    'theta' is the polar angle (inclination) between the +z axis and the vector from origin to p, normalized to [0,1].
    'phi' is the azimuthal angle from the +x axis in the xy-plane, normalized to [0,1].
    \param[in] p Cartesian coordinates (x,y,z).
    \return Spherical coordinates (theta,phi).
*/
float2 NrcCartesianToSphericalUnorm(float3 p)
{
    const float kOoneOverPi = 0.318309886183790671538; // 1/pi
    const float kOneOverTwoPi = 0.159154943091895335769; // 1/2pi

    p = normalize(p);
    float2 sph;
    sph.x = acos(p.z) * kOoneOverPi;
    sph.y = atan2(-p.y, -p.x) * kOneOverTwoPi + 0.5f;
    return sph;
}

/** Safer version of 'NrcCartesianToSphericalUnorm' which encodes infinities to zeros
 */
float2 NrcSafeCartesianToSphericalUnorm(float3 p)
{
    if (any(!isfinite(p)))
    {
        return float2(0.0f, 0.0f);
    }
    return NrcCartesianToSphericalUnorm(p);
}

// Built-in isnan() and isinf() need special compiler option and might be optimized away. Let's use these explicit checks instead.

/** Check if input value is a NaN
    \param[in] x Value to check.
    \return Return true if value is NaN
*/
bool NrcIsNan(float x)
{
    return (asuint(x) & 0x7FFFFFFF) > 0x7F800000;
}

/** Check if input value is an Inf
    \param[in] x Value to check.
    \return Return true if value is Inf
*/
bool NrcIsInf(float x)
{
    return (asuint(x) & 0x7FFFFFFF) == 0x7F800000;
}

/** Check if input vector has a NaN
    \param[in] v Vector to check.
    \return Return true if any of vector's components is a NaN
*/
bool NrcIsNan(float3 v)
{
    return (NrcIsNan(v.x) || NrcIsNan(v.y) || NrcIsNan(v.z));
}

/** Check if input vector has an Inf
    \param[in] v Vector to check.
    \return Return true if any of vector's components is an Inf
*/
bool NrcIsInf(float3 v)
{
    return (NrcIsInf(v.x) || NrcIsInf(v.y) || NrcIsInf(v.z));
}

/** Check if input vector contains either Infs or NaNs and output zeros if it does
    \param[in] input Vector to sanitize.
    \return Sanitized input vector free of NaNs and Infs
*/
float3 NrcSanitizeNansInfs(float3 input)
{
    if (NrcIsInf(input) || NrcIsNan(input))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return input;
}

/** Maps path information (QueryPathInfo) for given pixel coordinates to the correct position in the buffer
    \param[in] frameDimensions Frame Dimensions (width,height).
    \param[in] pixel Pixel Coordinates (x,y).
    \param[in] sampleIndex Index of the sample (path).
    \param[in] samplesPerPixel Number of samples (paths) per pixel.
    \return Index into QueryPathInfo buffer.
*/
uint NrcCalculateQueryPathIndex(const uint2 frameDimensions, const uint2 pixel, const uint sampleIndex, const uint samplesPerPixel)
{
    // Index in a linearized Tensor: (height, width, spp)
    uint index = sampleIndex;
    uint stride = samplesPerPixel;
    index += pixel.x * stride;
    stride *= frameDimensions.x;
    index += pixel.y * stride;
    return index;
}

/** Maps path information (TrainingPathInfo) for given pixel coordinates to the correct position in the buffer
    \param[in] trainingDimensions Training Dimensions (width,height).
    \param[in] pixel Pixel Coordinates (x,y).
    \param[in] sampleIndex Index of the sample (path).
    \param[in] samplesPerPixel Number of samples (paths) per pixel.
    \return Index into QueryPathInfo buffer.
*/
uint NrcCalculateTrainingPathIndex(const uint2 trainingDimensions, const uint2 pixel)
{
    return (trainingDimensions.x * pixel.y) + pixel.x;
}

/** Maps path vertex structure (PathVertex) for given pixel coordinates and path segment to the correct position in the Path Vertex buffer
    \param[in] trainingDimensions Training resolution (width,height).
    \param[in] pixel Pixel Coordinates (x,y).
    \param[in] vertexIdx Index of the vertex (ray segment) within the light path.
    \param[in] maxPathVertices Maximal number of light bounces allowed per path.
    \return Index into PathVertex buffer.
*/
uint NrcCalculateTrainingPathVertexIndex(uint2 trainingDimensions, uint2 pixel, const uint vertexIdx, const uint maxPathVertices)
{
    uint trainingPathIndex = NrcCalculateTrainingPathIndex(trainingDimensions, pixel);
    return trainingPathIndex * maxPathVertices + vertexIdx;
}

/** Updates radiance and throughput stored in the packed PathVertex structure.
    \param[in] packedPathVertex Packed PathVertex structure to update.
    \param[in] radiance New radiance value (r, g, b).
    \param[in] throughput New throughput value (r, g, b).
    \return Updated packed PathVertex structure.
*/
NrcPackedPathVertex NrcUpdateTrainingPathVertex(NrcPackedPathVertex packedPathVertex, const float3 radiance, const float3 throughput)
{
    packedPathVertex.data[0] = NrcEncodeLogLuvHdr(NrcSanitizeNansInfs(radiance));
    packedPathVertex.data[1] = NrcEncodeLogLuvHdr(NrcSanitizeNansInfs(throughput));

    return packedPathVertex;
}

/** Unpacks PathVertex structure, using provided radiance and throughput instead of the stored ones
    \param[in] packed Packed PathVertex structure.
    \param[in] radiance New radiance value (r, g, b).
    \param[in] throughput New throughput value (r, g, b).
    \return Unpacked PathVertex structure.
*/
NrcPathVertex NrcUnpackPathVertex(const NrcPackedPathVertex packed, const float3 radiance, const float3 throughput)
{
    NrcPathVertex vertex;
    vertex.radiance = NrcSanitizeNansInfs(radiance);
    vertex.throughput = NrcSanitizeNansInfs(throughput);
    vertex.linearRoughness = asfloat(packed.data[2]);
    vertex.normal = NrcDecodeNormal2x16(packed.data[3]);
    vertex.viewDirection = NrcDecodeNormal2x16(packed.data[4]);
    vertex.albedo = NrcUnpackR11G11B10(packed.data[5]);
    vertex.specular = NrcUnpackR11G11B10(packed.data[6]);

    vertex.encodedPosition = packed.encodedPosition;

    return vertex;
}

/** Unpacks PathVertex structure
    \param[in] packed Packed PathVertex structure.
    \return Unpacked PathVertex structure.
*/
NrcPathVertex NrcUnpackPathVertex(const NrcPackedPathVertex packed)
{
    NrcPathVertex vertex;
    vertex.radiance = NrcDecodeLogLuvHdr(packed.data[0]);
    vertex.throughput = NrcDecodeLogLuvHdr(packed.data[1]);
    vertex.linearRoughness = asfloat(packed.data[2]);
    vertex.normal = NrcDecodeNormal2x16(packed.data[3]);
    vertex.viewDirection = NrcDecodeNormal2x16(packed.data[4]);
    vertex.albedo = NrcUnpackR11G11B10(packed.data[5]);
    vertex.specular = NrcUnpackR11G11B10(packed.data[6]);

    vertex.encodedPosition = packed.encodedPosition;

    return vertex;
}

/** Increments the specified counter by one
    \param[in] countersData Counter buffer to update.
    \param[in] counterID ID (index) of the counter to update.
    \return Original counter value before the increment.
*/
uint NrcIncrementCounter(NRC_RW_STRUCTURED_BUFFER(uint) countersData, NrcCounter counter)
{
#if 1 //< Optimized version with one InterlockedAdd per warp. This can be always on

    uint laneCount = WaveActiveCountBits(true);
    uint laneOffset = WavePrefixCountBits(true);
    uint originalValue;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(countersData[(uint)counter], laneCount, originalValue);
    }
    originalValue = WaveReadLaneFirst(originalValue); // Broadcast to all active threads
    return originalValue + laneOffset;
#else //< Reference implementation with one InterlockedAdd per thread
    uint originalValue;
    InterlockedAdd(countersData[(uint)counter], 1, originalValue);
    return originalValue;
#endif
}

/** Encode a world space position to NRC's internal format.
    This version of the function can be used for games where a world space position
    can be reliably expressed as fp32 fields
    \param[in] worldSpacePosition - the world space position
    \param[in] nrcConstants
*/
NrcEncodedPosition NrcEncodePosition(float3 worldSpacePosition, NrcConstants nrcConstants)
{
#if TCNN_USES_FIXED_POINT_POSITIONS
    return NrcEncodedPosition(int3(worldSpacePosition * nrcConstants.scenePosScale)) + nrcConstants.scenePosBias;
#else
    return mad(worldSpacePosition, nrcConstants.scenePosScale, nrcConstants.scenePosBias);
#endif
}

/** Encode a world space position to NRC's internal format.
    This version of the function can be used for games that use large worlds where
    positions can't reliably be represented in a single fp32.
    Those games often have a 'local origin' using numbers that can be accurately
    stored in fp32, and a 'local offset'.
    \param[in] localPositionOffset - position relative to local origin
    \param[in] localOrigin - world space local origin
    \param[in] nrcConstants
*/
NrcEncodedPosition NrcEncodePosition(float3 localPositionOffset, float3 localOrigin, NrcConstants nrcConstants)
{
#if TCNN_USES_FIXED_POINT_POSITIONS
    // Convert incoming origin to fixed-point format
    NrcEncodedPosition encodedPosition = NrcEncodedPosition(int3(localOrigin * nrcConstants.scenePosScale)) + nrcConstants.scenePosBias;

    // To apply the offset, we just need to scale
    encodedPosition += NrcEncodedPosition(int3(localPositionOffset * nrcConstants.scenePosScale));

    return encodedPosition;
#else
    // This won't provide enough precision if the local origin gets too far from
    // the world origin, but this code path hopefully won't exist for too long
    return mad(localOrigin + localPositionOffset, nrcConstants.scenePosScale, nrcConstants.scenePosBias);
#endif
}

/** Creates the RadianceParams structure from data stored in the vertex.
    Normal and view direction is packed using spherical coordinates
    \param[in] vertex PathVertex structure to unpack
    \return RadianceParams corresponding to given input vertex.
*/
NrcRadianceParams NrcCreateRadianceParams(const NrcPathVertex vertex)
{
    NrcRadianceParams params;
    params.encodedPosition = vertex.encodedPosition;
    params.roughness = vertex.linearRoughness;
    params.normal = NrcSafeCartesianToSphericalUnorm(vertex.normal);
    params.viewDirection = NrcSafeCartesianToSphericalUnorm(vertex.viewDirection);
    params.albedo = vertex.albedo;
    params.specular = vertex.specular;
    return params;
}

/** Initializes packed PathVertex structure. Parameters are encoded, radiance is initialize to zero and throughput to one.
    \param[in] linearRoughness Roughness of the hit surface
    \param[in] shadingNormal Shading normal. If unavailable, a geometry normal should be used instead (x, y, z).
    \param[in] viewDirection Direction towards the viewer or opposite direction of the incident ray (x, y, z).
    \param[in] diffuseAlbedo Diffuse albedo of the hit surface (r, g, b).
    \param[in] specularAlbedo Specular albedo of the hit surface (r, g, b).
    \param[in] position Position of the hit (x, y, z).
    \return Encoded and packed PathVertex structure.
*/
NrcPackedPathVertex NrcInitializePackedPathVertex(const float linearRoughness,
                                                  const float3 shadingNormal,
                                                  const float3 viewDirection,
                                                  const float3 diffuseAlbedo,
                                                  const float3 specularAlbedo,
                                                  const NrcEncodedPosition encodedPosition)
{
    NrcPackedPathVertex v;
    v.data[0] = NrcEncodeLogLuvHdr(float3(0.f, 0.f, 0.f));
    v.data[1] = NrcEncodeLogLuvHdr(float3(1.f, 1.f, 1.f));
    v.data[2] = asuint(linearRoughness);
    v.data[3] = NrcEncodeNormal2x16(shadingNormal);
    v.data[4] = NrcEncodeNormal2x16(viewDirection);
    v.data[5] = NrcPackR11G11B10(diffuseAlbedo);
    v.data[6] = NrcPackR11G11B10(specularAlbedo);

    v.encodedPosition = encodedPosition;

    v.pad0 = 0;
    v.pad1 = 0;

    return v;
}

/** Packs and encodes PathInfo structure into smaller footprint.
    Note that fields 'vertexCount' and 'queryIndex' are encoded using 8 bits, limiting their range to [0;255]!
    \param[in] pathInfo PathInfo structure to pack
    \return Encoded and packed PathInfo structure.
*/
NrcPackedTrainingPathInfo NrcPackTrainingPathInfo(const NrcTrainingPathInfo pathInfo)
{
    NrcPackedTrainingPathInfo packedPathInfo = (NrcPackedTrainingPathInfo)0;

    packedPathInfo.packedData = pathInfo.packedData;
    packedPathInfo.queryBufferIndex = pathInfo.queryBufferIndex;

    return packedPathInfo;
}

/** Unpacks TrainingPathInfo structure
    \param[in] packedPathInfo PathInfo structure to unpack
    \return Unpacked TrainingPathInfo structure.
*/
NrcTrainingPathInfo NrcUnpackTrainingPathInfo(const NrcPackedTrainingPathInfo packedPathInfo)
{
    NrcTrainingPathInfo pathInfo = (NrcTrainingPathInfo)0;

    pathInfo.packedData = packedPathInfo.packedData;
    pathInfo.queryBufferIndex = packedPathInfo.queryBufferIndex;

    return pathInfo;
}

/** Unpacks PathInfo structure
    \param[in] packedPathInfo PathInfo structure to unpack
    \return Unpacked PathInfo structure.
*/
NrcQueryPathInfo NrcUnpackQueryPathInfo(const NrcPackedQueryPathInfo packedPathInfo)
{
    NrcQueryPathInfo pathInfo = (NrcQueryPathInfo)0;

    pathInfo.prefixThroughput = NrcDecodeLogLuvHdr(packedPathInfo.prefixThroughput);
    pathInfo.queryBufferIndex = packedPathInfo.queryBufferIndex;

    return pathInfo;
}

/** Unpacks radiance values from the buffer containing the query results

    The radiance values that go into NRC are re-scaled into a friendly range
    according to whatever was set in maxExpectedAverageRadianceValue.
    This function puts them back into the correct value range for the application.
    If you're using the built-in resolve pass, then you won't need to worry about
    this. You'll only need to use this if you use a custom resolve pass when
    you're picking up the query results straight from NRC.
*/
float3 NrcUnpackQueryRadiance(const NrcConstants nrcConstants, float3 packedQueryRadiance)
{
    return packedQueryRadiance * nrcConstants.radianceUnpackMultiplier;
}

#endif
#endif // __NRC_HELPERS_HLSL__
