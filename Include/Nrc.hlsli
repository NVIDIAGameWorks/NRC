/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __NRC_HLSLI__
#define __NRC_HLSLI__
// clang-format off

// -------------------------------------------------------------------------
// Handle NRC preprocessor macros
// -------------------------------------------------------------------------

// NRC can be configured to be in Update or Query modes, or can be
// entirely disabled. You can control the behaviour using one of
// these mechanisms:
//
// define ENABLE_NRC=1, but leave NRC_QUERY and NRC_UPDATE undefined
//     Doing this enables you to control whether NRC is in Update
//     or Query mode by setting g_nrcMode at the start of your shader.
//     This effectively makes it a compile-time choice, but enables
//     you to have both Update and Query raygen shaders compiled
//     as part of the same library if you wish.
//
// define either NRC_QUERY=1 or NRC_UPDATE=1
//     Doing this will force NRC into either Update or Query mode.
//
// define ENABLE_NRC=0, or leave it undefined
//     Doing this will effectively disable NRC
//
// Whichever option you choose, you can test the mode with the following
// functions
//     NrcIsEnabled()
//     NrcIsUpdateMode()
//     NrcIsQueryMode()

enum class NrcMode
{
    Disabled,
    Update,
    Query
};

#if defined(ENABLE_NRC)
// Validate
#if !((ENABLE_NRC == 0) || (ENABLE_NRC == 1))
#error "If you #define ENABLE_NRC, please set it to 0 or 1 to disable or enable NRC respectively"
#endif
#else
// Automatically set ENABLE_NRC as appropriate
#if (defined(NRC_UPDATE) || defined(NRC_QUERY))
#define ENABLE_NRC 1
#else
#define ENABLE_NRC 0
#endif
#endif

// Validate defines
#if !defined(ENABLE_NRC)
#error "Expected ENABLE_NRC to be defined to something"
#endif
#if ENABLE_NRC && !(!(defined(NRC_UPDATE)) || !(defined(NRC_QUERY))) 
#error "ENABLE_NRC=1, so expect to have one of NRC_UPDATE or NRC_QUERY, or neither defined"
#endif
#if !ENABLE_NRC && (defined(NRC_UPDATE) || defined(NRC_QUERY))
#if(NRC_UPDATE || NRC_QUERY)
#error "ENABLE_NRC=0, so expect to have neither NRC_UPDATE or NRC_QUERY"
#endif
#endif

#if (defined(NRC_UPDATE) || defined(NRC_QUERY))

// Here, the app has kindly declared the NRC_UPDATE or NRC_QUERY macros
// so we can configure the NrcMode as a compile-time const
#if defined(NRC_UPDATE)
#if defined(NRC_QUERY)
#if NRC_UPDATE == NRC_QUERY
#error "NRC_UPDATE and NRC_QUERY are mutually exclusive. Please only #define one of them to 1."
#endif
#else
#define NRC_QUERY (!NRC_UPDATE)
#endif
#else
#define NRC_UPDATE (!NRC_QUERY)
#endif

#if NRC_UPDATE
static const NrcMode g_nrcMode = NrcMode::Update;
#else
static const NrcMode g_nrcMode = NrcMode::Query;
#endif

#elif ENABLE_NRC

// ENABLE_NRC is enabled, but neither NRC_UPDATE or NRC_QUERY are
// defined, so we allow the NRC mode to be controlled by the app setting this
// only once in the shader (effectively making it compile-time), or even allow
// it to be controlled at run-time (not recommended).
static NrcMode g_nrcMode = NrcMode::Disabled;

#else

// Disable NRC
#define NRC_QUERY 0
#define NRC_UPDATE 0
static const NrcMode g_nrcMode = NrcMode::Disabled;

#endif

#if !defined(NRC_USE_CUSTOM_BUFFER_ACCESSORS)
#define NRC_USE_CUSTOM_BUFFER_ACCESSORS 0
#endif

// Allow the integration to define its own type for a read-write structured buffer
// E.g. if you are doing a Pirate Engine integration, the correct syntax is
//
//     Tharr be RW Structure Buffer <T> arrrrr
//
// So you would do
//
//     #define NRC_RW_STRUCTURED_BUFFER(T) Tharr be RW Structure Buffer <T> arrrrr
//
#if !defined(NRC_RW_STRUCTURED_BUFFER)
#define NRC_RW_STRUCTURED_BUFFER(T) RWStructuredBuffer<T>
#endif

// Allow engine to override declaring buffers in the `NrcBuffers` struct as part
// of the `NrcContext` in the case that buffers within structs is not supported.
// In order to do this, the engine must first
//
//     #define NRC_USE_CUSTOM_BUFFER_ACCESSORS 1
//
// and then it must declare the following macros to provide a read-write
// object that can be indexed with the square bracket operator
//
//     NRC_BUFFER_QUERY_PATH_INFO
//     NRC_BUFFER_TRAINING_PATH_INFO
//     NRC_BUFFER_TRAINING_PATH_VERTICES
//     NRC_BUFFER_QUERY_RADIANCE_PARAMS
//     NRC_BUFFER_QUERY_COUNTERS_DATA
//
#if ENABLE_NRC && !NRC_USE_CUSTOM_BUFFER_ACCESSORS
#if defined(NRC_BUFFER_QUERY_PATH_INFO) || defined(NRC_BUFFER_TRAINING_PATH_INFO) || defined(NRC_BUFFER_TRAINING_PATH_VERTICES) || defined(NRC_BUFFER_QUERY_RADIANCE_PARAMS) || defined(NRC_BUFFER_QUERY_COUNTERS_DATA) || defined(NRC_BUFFER_DEBUG_TRAINING_PATH_INFO)
#error "If you enable any NRC_BUFFER macros, then please #define NRC_USE_CUSTOM_BUFFER_ACCESSORS 1"
#endif
// Define the standard NRC buffer accessor macros
#define NRC_BUFFER_QUERY_PATH_INFO context.buffers.queryPathInfo
#define NRC_BUFFER_TRAINING_PATH_INFO context.buffers.trainingPathInfo
#define NRC_BUFFER_TRAINING_PATH_VERTICES context.buffers.trainingPathVertices
#define NRC_BUFFER_QUERY_RADIANCE_PARAMS context.buffers.queryRadianceParams
#define NRC_BUFFER_QUERY_COUNTERS_DATA context.buffers.countersData
#endif

#include "NrcHelpers.hlsli"

#ifndef __cplusplus

// Return true if Nrc is enabled.
// In nearly all cases, this will be compile-time evaluation.
bool NrcIsEnabled()
{
    return g_nrcMode != NrcMode::Disabled;
}

// Return true if Nrc is configured in Update mode.
// In nearly all cases, this will be compile-time evaluation.
bool NrcIsUpdateMode()
{
    return g_nrcMode == NrcMode::Update;
}

// Return true if Nrc is configured in Query mode.
// In nearly all cases, this will be compile-time evaluation.
bool NrcIsQueryMode()
{
    return g_nrcMode == NrcMode::Query;
}

enum class NrcProgressState
{
    Continue,
    TerminateImmediately,
    TerminateAfterDirectLighting,
};

#if ENABLE_NRC

// -------------------------------------------------------------------------
// Internal NRC Helpers
// -------------------------------------------------------------------------

/** Internal helper.
    Returns true when we termination heuristic is satisfied and we can terminate path and create the NRC query point
 */
bool NrcEvaluateTerminationHeuristic(const NrcPathState pathState, float threshold)
{
    return (pathState.primarySpreadRadius > 0.0f) && (pathState.cumulSpreadRadius > (threshold * pathState.primarySpreadRadius));
}

// Layout of pathState.packedData
//   +-----------+-------------+--------------+
//   | 15     12 | 11        7 | 6          0 |
//   +-----------+-------------+--------------+
//   |   Flags   | Termination |    Vertex    |
//   |           |   Reason    |     Count    |
//   +-----------+-------------+--------------+
static const NrcPackableUint nrcTerminationReasonShift          = 7;
static const NrcPackableUint nrcPathFlagsShift                  = 12;
static const NrcPackableUint nrcPathFlagHasExitedScene          = (NrcPackableUint) (1U << (nrcPathFlagsShift + 0U)); //< Training paths that exited the scene should not be "Q-learned"
static const NrcPackableUint nrcPathFlagIsUnbiased              = (NrcPackableUint) (1U << (nrcPathFlagsShift + 1U)); //< Some of the training paths are marked as "unbiased" to be extended through the entire scene
static const NrcPackableUint nrcPathFlagPreviousHitWasDeltaLobe = (NrcPackableUint) (1U << (nrcPathFlagsShift + 2U));
static const NrcPackableUint nrcPathFlagHeuristicReset          = (NrcPackableUint) (1U << (nrcPathFlagsShift + 3U));
static const NrcPackableUint nrcVertexCountMask                 = (NrcPackableUint) ((1U << nrcTerminationReasonShift) - 1);
static const NrcPackableUint nrcTerminationReasonMask           = (NrcPackableUint) (((1U << nrcPathFlagsShift) - 1U) & ~nrcVertexCountMask);

void NrcSetFlag(inout NrcPackableUint packedData, in NrcPackableUint flag)
{
    packedData |= flag;
}

void NrcClearFlag(inout NrcPackableUint packedData, in NrcPackableUint flag)
{
    packedData &= ~flag;
}

void NrcSetFlag(inout NrcPackableUint packedData, in NrcPackableUint flag, in bool value)
{
    packedData &= ~flag;
    packedData |= value ? flag : 0;
}

bool NrcGetFlag(in NrcPackableUint packedData, in NrcPackableUint flag)
{
    return (packedData & flag) ? true : false;
}

uint NrcGetVertexCount(in NrcPackableUint packedData)
{
    return (packedData & nrcVertexCountMask);
}

void NrcSetVertexCount(inout NrcPackableUint packedData, uint vertexCount)
{
    packedData &= ~nrcVertexCountMask;
    packedData |= (NrcPackableUint) vertexCount;
}

// -------------------------------------------------------------------------
// Public NRC Shader API
// -------------------------------------------------------------------------

/** When a path is terminated, call this to specify the reason.
    Used for debugging.
*/
void NrcSetDebugPathTerminationReason(inout NrcPathState pathState, NrcDebugPathTerminationReason reason)
{
    pathState.packedData &= ~nrcTerminationReasonMask;
    pathState.packedData |= ((NrcPackableUint) reason) << nrcTerminationReasonShift;
}

NrcDebugPathTerminationReason NrcGetDebugPathTerminationReason(in NrcPathState pathState)
{
    return (NrcDebugPathTerminationReason) ((pathState.packedData & nrcTerminationReasonMask) >> nrcTerminationReasonShift);
}

/** Creates a new NrcContext
    \param[in] constants. NrcConstants passed from the Nrc SDK library.
    \param[in] buffers. NrcBuffers struct that should be filled in by the app.
    \param[in] pixelIndex. The pixel coordinate.
*/
NrcContext NrcCreateContext(in NrcConstants constants, in NrcBuffers buffers, in uint2 pixelIndex)
{
    NrcContext context;
    context.constants = constants;
    context.buffers = buffers;
    context.pixelIndex = pixelIndex;
    context.sampleIndex = 0;

    return context;
}

/** Creates a fresh NrcPathState for a new path.
    Call this before entering the bounce loop.
    \param[in] constants. The NRC constants.
    \param[in] rand0to1. A random number between 0 and 1.
*/
NrcPathState NrcCreatePathState(in NrcConstants constants, float rand0to1)
{
    NrcPathState pathState = (NrcPathState) 0;
    pathState.queryBufferIndex = 0xFFFFFFFF;
    pathState.packedPrefixThroughput = 0;
    pathState.cumulSpreadRadius = 0.0f;
    pathState.primarySpreadRadius = 0.0f;
    pathState.packedData = 0;

    // Get a pseudorandom selection of "unbiased" training paths. Unbiased means that the paths are traced to their full length.
    const bool isUnbiased = NrcIsUpdateMode() && (rand0to1 < constants.proportionUnbiased);
    NrcSetFlag(pathState.packedData, nrcPathFlagIsUnbiased, isUnbiased);

    return pathState;
}

/** Sets the sample index of the path that we're going to trace.
    \param[inout] context
    \param[in]    sampleIndex. Optional sample index when rendering multiple paths per pixel
*/
void NrcSetSampleIndex(inout NrcContext context, in uint sampleIndex)
{
    context.sampleIndex = sampleIndex;
}

/** Determines whether application can use termination by russian roulette for this path.
    RR should not be used when we need to trace long unbiased paths to train the NRC.

    \param[in] pathState. NrcPathState structure created during path tracing
    \return Returns true when application can use russian roulette (RR) for this path.
*/
bool NrcCanUseRussianRoulette(in NrcPathState pathState)
{
    return !NrcGetFlag(pathState.packedData, nrcPathFlagIsUnbiased);
}

/** This should be called when the path traced ray segment is a 'hit'.
    Note that throughput and radiance will be modified by this function when NRC
    is in Update mode.  This is because NRC needs to know the throughput and direct
    light for each path segment.

    \param[in] context. NrcContext.
    \param[inout] pathState. NrcPathState to be updated.
    \param[in] surfaceAttributes. Information about the surface that was hit.
    \param[in] hitDistance. Distance from the previous hit, or the path length so far if this is
                            the first call with bounce > 0 when using Primary Surface Replacement.
    \param[in] bounce. The index of this bounce (the primary hit is bounce 0)
    \param[inout] throughput. The path tracer's accumulated throughput.
    \param[inout] radiance. The path tracer's accumulated radiance.
    \return Returns NrcProgressState to determine if and when this path should be terminated.
*/
NrcProgressState NrcUpdateOnHit(
    in NrcContext context,
    inout NrcPathState pathState,
    NrcSurfaceAttributes surfaceAttributes,
    float hitDistance,
    uint bounce,
    inout float3 throughput,
    inout float3 radiance)
{
    if (!NrcIsEnabled())
    {
        return NrcProgressState::Continue;
    }

    // Update the path spread approximation, used to trigger the path termination heuristic.
    // The heuristic prevents querying the NRC before the signal has been sufficiently blurred by the path spread.
    // This needs to be calculated even when heuristics is disabled, because it's still used for training paths
    const float cosGamma = abs(dot(surfaceAttributes.viewVector, surfaceAttributes.shadingNormal));
    if (pathState.primarySpreadRadius == 0.f)
    {
        const float kOneOverFourPI = 0.079577471545947667884f; // 1/4pi
        pathState.primarySpreadRadius = (NrcPackableFloat) (hitDistance / sqrt(cosGamma * kOneOverFourPI));
    }
    else if (!NrcGetFlag(pathState.packedData, nrcPathFlagPreviousHitWasDeltaLobe))
    {
        pathState.cumulSpreadRadius += (NrcPackableFloat) (hitDistance / sqrt(cosGamma * pathState.brdfPdf /* The BRDF PDF of the previous hit */));
    }
    NrcSetFlag(pathState.packedData, nrcPathFlagPreviousHitWasDeltaLobe, surfaceAttributes.isDeltaLobe);

    // Determine if we want to skip querying NRC at this bounce (e.g., we want skip mirrors)
    const bool skipVertex = (context.constants.skipDeltaVertices || context.constants.enableTerminationHeuristic) && surfaceAttributes.isDeltaLobe;
    if (skipVertex)
    {
        return NrcProgressState::Continue;
    }
    
    uint vertexCount = NrcGetVertexCount(pathState.packedData);
    if (NrcIsUpdateMode())
    {
        // Write training path vertex information
        const uint trainingPathVertexIndex = NrcCalculateTrainingPathVertexIndex(context.constants.trainingDimensions, context.pixelIndex, vertexCount, context.constants.maxPathVertices);
        if (vertexCount > 0)
        {
            // Finalize the previous vertex with the radiance and throughput that the path tracer accumulated
            // during its previous iteration
            const uint previousTrainingPathVertexIndex = trainingPathVertexIndex - 1;
            NRC_BUFFER_TRAINING_PATH_VERTICES[previousTrainingPathVertexIndex] = NrcUpdateTrainingPathVertex(NRC_BUFFER_TRAINING_PATH_VERTICES[previousTrainingPathVertexIndex], radiance, throughput);
        }

        // Always update vertex counts. The pathState vertexCount variable mostly mirrors 'bounce' variable,
        // but it does not count specular vertices if these were marked to be skipped.
        // This is needed to ensure that a surface scene in a mirror is handled similarly to surfaces seen directly.
        vertexCount++;
        NrcSetVertexCount(pathState.packedData, vertexCount);

        // Reset the path tracer's throughput and radiance for the next
        // path segment.
        throughput = 1..xxx;
        radiance = 0..xxx;

        // Store path vertex
        NRC_BUFFER_TRAINING_PATH_VERTICES[trainingPathVertexIndex] = NrcInitializePackedPathVertex(
            surfaceAttributes.roughness, surfaceAttributes.shadingNormal, surfaceAttributes.viewVector, surfaceAttributes.diffuseReflectance, surfaceAttributes.specularF0, surfaceAttributes.encodedPosition);

        bool terminate = (bounce == context.constants.maxPathVertices - 1); //< Is this path at last vertex already? If yes, we can terminate.
        if(!NrcGetFlag(pathState.packedData, nrcPathFlagIsUnbiased))
        {
            if(NrcEvaluateTerminationHeuristic(pathState, context.constants.trainingTerminationHeuristicThreshold))
            {
                // We should run the path to its normal termination, then reset the spread radius and run again
                // until we hit the termination criteria a second time
                terminate |= NrcGetFlag(pathState.packedData, nrcPathFlagHeuristicReset);
                NrcSetFlag(pathState.packedData, nrcPathFlagHeuristicReset);
                pathState.cumulSpreadRadius = 0.f;
            }
        }

        if( terminate )
        {
            return context.constants.radianceCacheDirect ? NrcProgressState::TerminateImmediately : NrcProgressState::TerminateAfterDirectLighting;
        }
    }
    else
    {
        // Do the kind of things that were in NRC_CreateQuery here
        //......
        
        // Check if we can query the cache at the current vertex (terminating the path)
        bool createQuery = false;
        if (context.constants.enableTerminationHeuristic)
        {
            // This evaluates more complex heuristic based on the spread of the ray cone approximating the ray along the path
            createQuery = NrcEvaluateTerminationHeuristic(pathState, context.constants.terminationHeuristicThreshold);
        }
        else 
        {
            // Termination criterion enabling debug visualization of the cache by querying at vertex index zero.
            createQuery = vertexCount == 0;
        }

        // Always update vertex counts. The pathState vertexCount variable mostly mirrors 'bounce' variable,
        // but it does not count specular vertices if these were marked to be skipped.
        // This is needed to ensure that a surface scene in a mirror is handled similarly to surfaces seen directly.
        vertexCount++;
        NrcSetVertexCount(pathState.packedData, vertexCount);

        // Create query record
        if (createQuery)
        {
            float3 prefixThroughput = (context.constants.learnIrradiance) ? (throughput * (surfaceAttributes.specularF0 + surfaceAttributes.diffuseReflectance)) : throughput;
            prefixThroughput = max(0.0f, NrcSanitizeNansInfs(prefixThroughput));
            pathState.packedPrefixThroughput = NrcEncodeLogLuvHdr(prefixThroughput);

            pathState.queryBufferIndex = NrcIncrementCounter(NRC_BUFFER_QUERY_COUNTERS_DATA, NrcCounter::Queries);

            NrcRadianceParams params;
            params.encodedPosition = surfaceAttributes.encodedPosition;
            params.roughness = surfaceAttributes.roughness;
            params.normal = NrcSafeCartesianToSphericalUnorm(surfaceAttributes.shadingNormal);
            params.viewDirection = NrcSafeCartesianToSphericalUnorm(surfaceAttributes.viewVector);
            params.albedo = surfaceAttributes.diffuseReflectance;
            params.specular = surfaceAttributes.specularF0;

            NRC_BUFFER_QUERY_RADIANCE_PARAMS[pathState.queryBufferIndex] = params;

            // Terminate now if the cache already includes direct reflected radiance.
            // Otherwise, we will terminate later, after NEE and the scatter ray has been computed.
            if (context.constants.radianceCacheDirect)
            {
                NrcSetDebugPathTerminationReason(pathState, NrcDebugPathTerminationReason::CreateQueryImmediate);
                return NrcProgressState::TerminateImmediately;
            }
            else
            {
                NrcSetDebugPathTerminationReason(pathState, NrcDebugPathTerminationReason::CreateQueryAfterDirectLighting);
                return NrcProgressState::TerminateAfterDirectLighting;
            }
        }
    }
    return NrcProgressState::Continue;
}

/** This should be called when the path traced ray segment is a 'miss'.

    \param[inout] pathState. NrcPathState to be updated.
*/
void NrcUpdateOnMiss(inout NrcPathState pathState)
{
    NrcSetDebugPathTerminationReason(pathState, NrcDebugPathTerminationReason::PathMissExit);
    NrcSetFlag(pathState.packedData, nrcPathFlagHasExitedScene);
}

/** Inform NRC of the PDF of the BRDF.
    NRC uses the PDF of the BRDF for its termination heuristic.  A path tracer usually
    evaluates this when figuring out what direction to shoot the next ray.
    Call this function at that point to pass this information to NRC.

    \param[inout] pathState. NrcPathState structure to be updated.
    \param[in] brdfPdf. The PDF of the BRDF.
*/
void NrcSetBrdfPdf(inout NrcPathState pathState, in float brdfPdf)
{
    pathState.brdfPdf = brdfPdf;
}

/** Write out whatever information is required when the path is finished.
    Call this after the path tracer's bounce loop.

    \param[in] context. NrcContext.
    \param[inout] pathState. NrcPathState to be updated.
    \param[in] throughput. The final throughput.
    \param[in] radiance. The final radiance.
*/
void NrcWriteFinalPathInfo(in    NrcContext context,
                           inout NrcPathState pathState,
                           in    float3 throughput,
                           in    float3 radiance)
{
    if (!NrcIsEnabled())
    {
        return;
    }
    if (NrcIsUpdateMode())
    {
        // Training pass

        uint vertexCount = NrcGetVertexCount(pathState.packedData);
        // Only create cache query for self-training if the last vertex throughput is non-zero
        if (vertexCount > 0)
        {
            const uint vertexIndex = vertexCount - 1;
            const uint arrayIndex = NrcCalculateTrainingPathVertexIndex(
                context.constants.trainingDimensions, context.pixelIndex, vertexIndex, context.constants.maxPathVertices);
            NRC_BUFFER_TRAINING_PATH_VERTICES[arrayIndex] = NrcUpdateTrainingPathVertex(NRC_BUFFER_TRAINING_PATH_VERTICES[arrayIndex], radiance, throughput);

            // Create self-training records for _all_ training paths, including unbiased ones.
            // Without self-training, each training vertex position within the path would matter.
            // Vertices closer to the tail end would receive less indirect illumination, since there
            // are less following vertices, than those closer to the head.
            // An alternative would be to condition the network prediction on the vertex index, but
            // this complicates the task of the network.
            if (!NrcGetFlag(pathState.packedData, nrcPathFlagHasExitedScene) && (context.constants.maxPathVertices > 1)) // && !NrcGetFlag(pathState.packedData, nrcPathFlagIsUnbiased))
            {
                NrcPathVertex vertex = NrcUnpackPathVertex(NRC_BUFFER_TRAINING_PATH_VERTICES[arrayIndex], radiance, throughput);
                pathState.queryBufferIndex = NrcIncrementCounter(NRC_BUFFER_QUERY_COUNTERS_DATA, NrcCounter::Queries);

                NRC_BUFFER_QUERY_RADIANCE_PARAMS[pathState.queryBufferIndex] = NrcCreateRadianceParams(vertex);
            }
        }

        NrcTrainingPathInfo unpackedPathInfo = (NrcTrainingPathInfo) 0;
        unpackedPathInfo.packedData = pathState.packedData;
        unpackedPathInfo.queryBufferIndex = pathState.queryBufferIndex;

        const uint trainingPathIndex = NrcCalculateTrainingPathIndex(context.constants.trainingDimensions, context.pixelIndex);
        NRC_BUFFER_TRAINING_PATH_INFO[trainingPathIndex] = NrcPackTrainingPathInfo(unpackedPathInfo);

    }
    else
    {
        // Query pass

        const uint queryPathIndex = NrcCalculateQueryPathIndex(context.constants.frameDimensions, context.pixelIndex, context.sampleIndex, context.constants.samplesPerPixel);

        NrcPackedQueryPathInfo packedQueryPathInfo = (NrcPackedQueryPathInfo) 0;

        // The prefix throughput was saved separately in case
        // radianceCacheDirect is set to false, in which case path throughput would also include
        // the BSDF weight of the query vertex, not just the prefix throughput.
        packedQueryPathInfo.prefixThroughput = pathState.packedPrefixThroughput;
        packedQueryPathInfo.queryBufferIndex = pathState.queryBufferIndex;

        NRC_BUFFER_QUERY_PATH_INFO[queryPathIndex] = packedQueryPathInfo;
    }
}

#else // !ENABLE_NRC

//
// Define stub functions for when NRC is disabled so that the caller does not
// need to guard code with preprocessor macros if they don't want to.
//

void NrcSetDebugPathTerminationReason(NrcPathState, NrcDebugPathTerminationReason)
{}

NrcContext NrcCreateContext(NrcConstants, NrcBuffers, uint2)
{
    NrcContext context;
    return context;
}

NrcPathState NrcCreatePathState(NrcConstants, float)
{
    NrcPathState pathState;
    return pathState;
}

void NrcSetSampleIndex(NrcContext, uint)
{}

bool NrcCanUseRussianRoulette(NrcPathState)
{
    return true;
}

NrcProgressState NrcUpdateOnHit(NrcContext, NrcPathState, NrcSurfaceAttributes, float, uint, float3, float3)
{
    return NrcProgressState::Continue;
}

void NrcUpdateOnMiss(NrcPathState)
{}

void NrcSetBrdfPdf(NrcPathState, float)
{}

void NrcWriteFinalPathInfo(NrcContext, NrcPathState, float3, float3)
{}

#endif // !__cplusplus

#endif // !ENABLE_NRC

#endif // __NRC_HLSLI__
