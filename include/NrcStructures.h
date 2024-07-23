/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __NRC_STRUCTURES_HLSL__
#define __NRC_STRUCTURES_HLSL__
#pragma once

// If this macro wasn't set by the user, set it to the default here
#ifndef NRC_PACK_PATH_16BITS
#define NRC_PACK_PATH_16BITS 0
#endif

// -------------------------------------------------------------------------
//    C++ compatibility
// -------------------------------------------------------------------------

// Set to 0 if TCNN is expecting UNORM positions in fp32 format
// Set to 1 if TCNN is expecting UNORM positions in 0:32 fixed point format
// Once we've fixed the issues with the fixed point format, we'll commit
// to that and delete all the code that uses the 0 side of this macro
// and delete this macro itself
#define TCNN_USES_FIXED_POINT_POSITIONS 0

#ifdef __cplusplus
typedef uint16_t nrc_float16_t;
typedef uint32_t nrc_float16_t2;
typedef uint32_t nrc_uint;
struct nrc_uint2
{
    uint32_t x, y;
};
struct nrc_uint3
{
    uint32_t x, y, z;
};
struct nrc_float2
{
    float x, y;
};
struct nrc_float3
{
    float x, y, z;
};
struct nrc_float4
{
    float x, y, z, w;
};
#else
typedef float16_t nrc_float16_t;
typedef float16_t2 nrc_float16_t2;
typedef uint nrc_uint;
typedef uint2 nrc_uint2;
typedef uint3 nrc_uint3;
typedef float2 nrc_float2;
typedef float3 nrc_float3;
typedef float4 nrc_float4;

#if NRC_PACK_PATH_16BITS
typedef uint16_t NrcPackableUint;
typedef float16_t NrcPackableFloat;
#else
typedef uint32_t NrcPackableUint;
typedef float NrcPackableFloat;
#endif

#endif

#if TCNN_USES_FIXED_POINT_POSITIONS
typedef nrc_uint3 NrcEncodedPosition;
#else
typedef nrc_float3 NrcEncodedPosition;
#endif

/** Enumeration of the atomic counters used.
 */
enum class NrcCounter
{
    Queries = 0,
    TrainingRecords = 1,

    // Must be last
    Count
};

/**
 *  Additional information about radiance evaluated/cached at certain point
 */
struct NrcRadianceParams
{
    NrcEncodedPosition encodedPosition;
    float roughness;
    nrc_float2 normal; // Shading normal. If unavailable, a geometry normal should be used instead
    nrc_float2 viewDirection; // Direction towards the viewer or opposite direction of the incident ray

    nrc_float3 albedo; // Diffuse albedo of the hit surface
    nrc_float3 specular; // Specular albedo of the hit surface
};

/**
 *  Information about the path being traced, needed to reconstruct the path and resolve radiance to create the final image
 */
struct NrcTrainingPathInfo
{
    uint32_t vertexCount;
    uint32_t queryBufferIndex;
    bool hasExitedScene;
};

/**
 *  Packed version of NrcTrainingPathInfo
 */
struct NrcPackedTrainingPathInfo
{
    // TODO: We can almost certainly get this down to a single uint32_t now
    uint32_t flagsAndIndices;
    uint32_t queryBufferIndex;
};

/**
 *  Information about the path being traced, needed to resolve radiance to create the final image
 */
struct NrcQueryPathInfo
{
    nrc_float3 prefixThroughput;
    uint32_t queryBufferIndex;
};

/**
 *  Packed version of NrcTrainingPathInfo
 */
struct NrcPackedQueryPathInfo
{
    uint32_t prefixThroughput;
    uint32_t queryBufferIndex;
};

/**
 *  Struct holding path vertex data for training the NRC.
 */
struct NrcPathVertex
{
    nrc_float3 radiance; ///< Reflected radiance
    nrc_float3 throughput; ///< Throughput to the next vertex

    NrcEncodedPosition encodedPosition; // nrc_float3 unormPosition; ///< World space position squashed into unorm range
    float linearRoughness; ///< Material roughness

    nrc_float3 normal; ///< Sampled direction
    nrc_float3 viewDirection; ///< Direction towards the previous path vertex

    nrc_float3 albedo; ///< Base diffuse reflectance
    nrc_float3 specular; ///< Base specular reflectance
};

/**
 *  Packed version of NrcPathVertex
 */
struct NrcPackedPathVertex
{
    uint32_t data[7];
    uint32_t pad0;

    NrcEncodedPosition encodedPosition;
    uint32_t pad1;
};

/**
 *  Debug Path Termination Reasons
 */
enum class NrcDebugPathTerminationReason
{
    Unset = 0,
    PathMissExit,
    CreateQueryImmediate,
    MaxPathVertices,
    CreateQueryAfterDirectLighting,
    RussianRoulette,
    BRDFAbsorption,
    COUNT
};

enum class NrcResolveMode
{
    // The default behaviour.
    // This takes the query result and adds it to the output buffer.
    AddQueryResultToOutput,

    // A debug mode that overwrites the output buffer with the query results
    ReplaceOutputWithQueryResult,

    // A debug mode that shows a heatmap for the number of training bounces.
    // You should see more bounces in corners, and from smooth surfaces.
    // How the number of vertices in the training path translates to colors:
    //           1 : Dark Red           ( 0.5, 0,   0   )
    //           2 : Bright Red         ( 1,   0,   0   )
    //           3 : Dark Yellow        ( 0.5, 0.5, 0   )
    //           4 : Green              ( 0,   1,   0   )
    //           5 : Dark Cyan          ( 0,   0.5, 0.5 )
    //           6 : Blue               ( 0,   0,   1   )
    //           7 : Bleugh (?)         ( 0.5, 0.5, 1   )
    // Miss or > 8 : White              ( 1,   1,   1   )
    TrainingBounceHeatMap,

    // The same as TrainingBounceHeatMap, but smoothed over time
    // to give a result more like you would see with accumulation.
    TrainingBounceHeatMapSmoothed,

    // A debug mode that shows the training radiance for the primary ray
    // segment.  This should look like a low resolution version of the path
    // traced result, and it will be noisy.
    PrimaryVertexTrainingRadiance,

    // The same as PrimaryVertexTrainingRadiance, but smoothed over time
    // to give a result more like you would see with accumulation
    PrimaryVertexTrainingRadianceSmoothed,

    // A debug mode that shows a random colour that's a hash of the query index.
    // When things are working correctly - this should look like coloured noise.
    QueryIndex,

    // Same as QueryIndex, but for the training pass's self-training records.
    // When things are working correctly - this should look like coloured noise.
    TrainingQueryIndex,

    // Direct visualization of the cache (equivalent of querying at vertex zero).
    // The recommended tool to assess correctness of integration, this debug view should
    // capture features such as shadows and view-dependent specular highlights and display
    // them in a low-detail, over-smoothed output.
    DirectCacheView,
};

/**
 *  Holds common parameters needed by NRC functions called from the path tracer.
 *  The app should fill this in using Context::PopulateShaderConstants and then
 *  pass it up to the path tracing shader, most likely in a constant buffer.
 */
struct NrcConstants
{
    nrc_uint2 frameDimensions;
    nrc_uint2 trainingDimensions;

    nrc_float3 scenePosScale;
    nrc_uint samplesPerPixel;

    NrcEncodedPosition scenePosBias;
    nrc_uint maxPathVertices;

    nrc_uint learnIrradiance;
    nrc_uint radianceCacheDirect;
    float radianceUnpackMultiplier; // See NrcUnpackQueryRadiance
    NrcResolveMode resolveMode;

    nrc_uint enableTerminationHeuristic;
    nrc_uint skipDeltaVertices;
    float terminationHeuristicThreshold;

    float trainingTerminationHeuristicThreshold;
    nrc_uint pad0;
    nrc_uint pad1;
};

struct NrcDebugTrainingPathInfo
{
    nrc_float3 primaryRadiance;
    nrc_float3 accumulation;
};

#ifndef __cplusplus
/**
 *  Attributes of the surface hit by the ray, needed to evaluate NRC query and training data
 */
struct NrcSurfaceAttributes
{
    NrcEncodedPosition encodedPosition; // Use NrcEncodePosition
    float roughness;
    nrc_float3 specularF0;
    nrc_float3 diffuseReflectance;
    nrc_float3 shadingNormal;
    nrc_float3 viewVector;
    bool isDeltaLobe;
};

/**
 *  Holds common parameters needed by NRC functions; called from the path tracer
 */
struct NrcBuffers
{
#if !NRC_USE_CUSTOM_BUFFER_ACCESSORS
    NRC_RW_STRUCTURED_BUFFER(NrcPackedQueryPathInfo) queryPathInfo;
    NRC_RW_STRUCTURED_BUFFER(NrcPackedTrainingPathInfo) trainingPathInfo;
    NRC_RW_STRUCTURED_BUFFER(NrcPackedPathVertex) trainingPathVertices;
    NRC_RW_STRUCTURED_BUFFER(NrcRadianceParams) queryRadianceParams;
    NRC_RW_STRUCTURED_BUFFER(uint) countersData;
#endif
};

/**
 *  Holds state data about path being traced.
 *  If your path tracer is split over multiple passes, or if you need to call NRC methods
 *  in a CHS shader - then NrcPathState is the struct that you need to store or pass around.
 *  Enable macro NRC_PACK_PATH_16BITS to pack certain fields into 16-bit types. This can be
 *  useful, e.g, when storing this structure in the ray payload to minimize its size.
 *
 *  Information for advanced users that might want to manually pack this structure
 *  and squeeze every bit of packing that they can:
 *
 *  packedPrefixThroughput and queryBufferIndex are used between the _final_ call to
 *  NrcUpdateOnHit and NrcWriteFinalPathInfo and they're not used at all during the update pass.
 *  So you don't need to store them if NrcUpdateOnHit and NrcWriteFinalPathInfo are
 *  in the same shader stage. And you don't need to store them in the update pass.
 *
 *  The maximum possible value for queryBufferIndex is
 *       (frameDimensions.x * frameDimensions.y * samplesPerPixel) +
 *       (trainingDimensions.x * trainingDimensions.y)
 *  So there might be some bits at the top that you could use.
 *
 *  primarySpreadRadius and cumulSpreadRadius are always positive, so the sign bit can be used.
 */
struct NrcPathState
{
    uint32_t packedPrefixThroughput;
    uint32_t queryBufferIndex;

    // Info for those that are packing this structure manually...
    // The remaining fields are used in update and query passes, so always need
    // to be stored when straddling shader stages.
    NrcPackableFloat primarySpreadRadius; ///< Approximated as `d^2 / cos` at primary hit.
    NrcPackableFloat cumulSpreadRadius; ///< Square root of the cumulative area spread at the current path vertex.

    NrcPackableUint packedData; ///< The number of vertices processed, flags and termination reason
    NrcPackableFloat brdfPdf;
};

/**
 *  Create an NrcContext with NrcCreateState.
 *  This structure is used by nearly all Nrc functions in nrc.hlsli
 */
struct NrcContext
{
    NrcConstants constants;
    NrcBuffers buffers;

    nrc_uint2 pixelIndex;
    uint32_t sampleIndex;
};
#endif // !__cplusplus

#endif // __NRC_STRUCTURES_HLSL__
