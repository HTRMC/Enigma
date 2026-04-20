// mp_cluster_layout.hlsl
// =======================
// Shared on-disk layout constants for micropoly pages — kept in one place so
// HLSL shaders cannot drift from src/asset/MpAssetFormat.h. Include-only.

#ifndef MICROPOLY_CLUSTER_LAYOUT_HLSL
#define MICROPOLY_CLUSTER_LAYOUT_HLSL

// ClusterOnDisk stride in bytes. Mirrors asset::kClusterOnDiskSize == 76.
#define MP_CLUSTER_ON_DISK_STRIDE 76u

// PagePayloadHeader size in bytes. Mirrors asset::kPagePayloadHeaderSize == 16.
#define MP_PAGE_PAYLOAD_HEADER_BYTES 16u

#endif // MICROPOLY_CLUSTER_LAYOUT_HLSL
