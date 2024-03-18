/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>

#include "NrcCommon.h"

struct NrcConstants;

namespace nrc
{
namespace vulkan
{
/**
 * Info about allocated buffer and its size
 */
struct BufferInfo
{
    VkBuffer resource = VK_NULL_HANDLE;
    size_t allocatedSize = 0;
    size_t allocatedOffset = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
};

/**
 *  All allocated NRC SDK buffers
 */
struct Buffers
{
    BufferInfo buffers[(int)BufferIdx::Count];

    // [] Operators to allow direct access to the array with BufferIdx
    const BufferInfo& operator[](BufferIdx idx) const
    {
        return buffers[(int)idx];
    }
    BufferInfo& operator[](BufferIdx idx)
    {
        return buffers[(int)idx];
    }
};

/**
 *  Get an array of Vulkan device extensions that are required for NRC.
 *  Returns the number of extensions that will be in the array.
 */
NRC_DECLSPEC uint32_t GetVulkanDeviceExtensions(char const* const*& outStringArray);

/**
 *  Get an array of Vulkan instance extensions that are required for NRC.
 *  Returns the number of extensions that will be in the array.
 */
NRC_DECLSPEC uint32_t GetVulkanInstanceExtensions(char const* const*& outStringArray);

/**
 *  Get an array of Vulkan device features that are required for NRC.
 *  Returns the number of features that will be in the array.
 */
NRC_DECLSPEC uint32_t GetVulkanDeviceFeatures(char const* const*& outStringArray);

/**
 *  Initialization of the NRC library.
 *  This should be done once at application startup, before any
 *  NRC Contexts are created
 */
NRC_DECLSPEC Status Initialize(const nrc::GlobalSettings& config);

/**
 *  Shutdown of the NRC library.
 *  This should be done once at application shutdown, after all
 *  NRC Contexts have been destroyed.
 */
NRC_DECLSPEC void Shutdown();

class Context
{
public:
    /**
     *  Create a new NRC Context.
     */
    NRC_DECLSPEC static Status Create(VkDevice device, VkPhysicalDevice gpu, VkInstance apiInstance, Context*& outContext);

    /**
     *  Destroys an NRC Context
     */
    NRC_DECLSPEC static Status Destroy(Context& context);

    /**
     *  Returns all information necessary to allocate GPU memory needed by SDK for a given ContextSettings.
     */
    NRC_DECLSPEC static Status GetBuffersAllocationInfo(const ContextSettings& contextSettings, BuffersAllocationInfo& outBuffersAllocationInfo);

    /**
     *  (Re)Configures the NRC Context, allocating or reallocating memory and buffers as required.
     *  Call this when something in the ContextSettings changes.  E.g. the resolution.
     *  This will also reload the JSON configuration if a filename was provided.
     *  You must call this at least once before starting to use any per-frame methods.
     *  This method performs memory allocations and initialisation.  Please consider that if
     *  this is used in your main render loop, it might cause a hitch.  (For example, if you
     *  were to re-configure the context in a dynamic resolution scaling scenario)
     *  You should pass Buffers if you have opted to manage the buffer allocations yourself
     *  (by setting enableGPUMemoryAllocation in the GlobalSettings to false)
     */
    NRC_DECLSPEC Status Configure(const ContextSettings& contextSettings, const Buffers* buffers = nullptr);

    /**
     *  Populate NrcConstants (shader constants structure).
     *  NrcConstants should be put into a constant buffer and passed to NRC_InitializeNRCParameters
     *  in your shaders.
     */
    NRC_DECLSPEC Status PopulateShaderConstants(NrcConstants& outConstants) const;

    /**
     *  This should be called at the beginning of the frame to allow NRC to do its necessary housekeeping.
     */
    NRC_DECLSPEC Status BeginFrame(VkCommandBuffer cmdList, const FrameSettings& frameSettings);

    /**
     *  Assumes the path tracer passes have created the training data and queries
     *  Input: TrainingRadiance, TrainingRadianceParams, Counters, CountersReadback,
     *         QueryRadianceParams, PathInfo, TrainingPathVertices
     *  Output: TrainingRadiance, TrainingRadianceParams, QueryRadiance, countersData
     *  Trains the neural cache to predict radiance values closer to the provided targets at given query points.
     *  Note: using 'trainingLossPtr' imposes a perf penalty. Set it to nullptr if unused.
     *  Enables the Neural Network to infer the amount of outgoing radiance for
     *  each queried vertex essentially allowing it to predict how much light should be injected
     *  at the end of the short paths.
     *  It also writes out training data which will later be used to optimize the Neural Network.
     */
    NRC_DECLSPEC Status QueryAndTrain(VkCommandBuffer cmdList, float* trainingLossPtr);

    /**
     *  The Resolve call uses path data and predicted radiance which will be used in the creation
     *  of the final image. It also writes out training data which will later be used to optimize
     *  the Neural Network.
     *  Input: QueryRadiance, PathInfo, TrainingPathVertices (assumes the path tracer pass wrote the entries)
     *  Output: TrainingRadiance, TrainingRadianceParams, countersData, outputBuffer
     */
    NRC_DECLSPEC Status Resolve(VkCommandBuffer cmdList, VkImageView outputBuffer);

    /**
     *  Trains the neural cache to predict radiance values closer to the provided targets at given query points.
     *  Note: using 'trainingLossPtr' imposes a perf penalty. Set it to nullptr if unused.
     *  Input: TrainingRadiance, TrainingRadianceParams, Counters, CountersReadback (assumes the resolve pass
     *  created training radiance parameters and incremented counters)
     */
    NRC_DECLSPEC Status Train(VkCommandBuffer cmdList, float* trainingLossPtr);

    /**
     *  Invoked after the command list has been submitted.
     *  The command queue must be the same one that was used to execute all the previous command lists.
     */
    NRC_DECLSPEC Status EndFrame(VkQueue cmdQueue);

    /**
     *  Returns buffer resources shared between NRC SDK and user application, such as buffers for storing training radiance and queries.
     */
    NRC_DECLSPEC const Buffers* GetBuffers() const;

    /**
     *  Returns pointer to the device used to initialize this NRC Context
     */
    NRC_DECLSPEC VkDevice GetDevice() const;

protected:
    // Prevent user code from being able to instanciate this class
    Context()
    {
    } // Please use nrc::Context::Create
    ~Context()
    {
    } // Please use nrc::Context::Destroy
};

} // namespace vulkan
} // namespace nrc

#undef EXPORTING_NRC
