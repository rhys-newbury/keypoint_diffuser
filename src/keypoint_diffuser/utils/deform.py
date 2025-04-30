import torch


def apply_general_deformation(
    point_cloud: torch.Tensor,
    max_stretch_factor: float = 1.2,
    max_bending_factor: float = 0.8,
    max_twist_factor: float = 0.9,
    max_taper_factor: float = 0.6,
    max_noise_std: float = 0.02,
    *,
    apply_stretch: bool = True,
    apply_bend: bool = True,
    apply_twist: bool = True,
    apply_taper: bool = True,
    apply_noise: bool = False,
):
    """
    Applies a range of deformations (stretch, bend, twist, taper, noise)
    using a transformation matrix.

    Args:
        point_cloud: (B, N, 3) tensor of point cloud positions.
        max_stretch_factor: Maximum range for stretch deformation.
        max_bending_factor: Maximum range for bending deformation.
        max_twist_factor: Maximum range for twist deformation.
        max_taper_factor: Maximum range for taper deformation.
        max_noise_std: Maximum standard deviation for noise.
        apply_stretch: Whether to apply stretching.
        apply_bend: Whether to apply bending.
        apply_twist: Whether to apply twisting.
        apply_taper: Whether to apply tapering.
        apply_noise: Whether to apply noise.

    Returns:
        deformed_cloud: (B, N, 3) tensor of deformed points.
        deformation_matrix: (B, 3, 3) transformation matrix.
    """
    batch_size, num_points, _ = point_cloud.shape
    device = point_cloud.device

    # Initialize identity deformation matrix (for affine transformations)
    deformation_matrix = torch.eye(3, device=device).repeat(batch_size, 1, 1)

    # Generate random transformation parameters for each batch
    stretch_factors = 1.0 + (
        torch.rand(batch_size, device=device) * (max_stretch_factor - 1.0)
    )
    bending_factors = torch.rand(batch_size, device=device) * max_bending_factor
    twist_factors = torch.rand(batch_size, device=device) * max_twist_factor
    taper_factors = torch.rand(batch_size, device=device) * max_taper_factor
    noise_std_factors = (
        torch.rand(batch_size, device=device) * max_noise_std if apply_noise else None
    )

    # 1. Stretching along X-axis
    if apply_stretch:
        deformation_matrix[:, 0, 0] = stretch_factors  # Stretch X-axis

    # 2. Bending (Modify Y-coordinates using sinusoidal function)
    if apply_bend:
        bending_offsets = bending_factors.view(batch_size, 1) * torch.sin(
            point_cloud[:, :, 0] * 3.14
        )
        deformation_matrix[:, 1, 0] = bending_offsets.mean(
            dim=1
        )  # Apply bending in Y direction

    # 3. Twisting (Rotation about X-axis)
    if apply_twist:
        theta = twist_factors.view(batch_size, 1) * point_cloud[:, :, 0].mean(
            dim=1, keepdim=True
        )  # Avg X for rotation
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        twist_matrix = torch.eye(3, device=device).repeat(batch_size, 1, 1)
        twist_matrix[:, 1, 1] = cos_theta.squeeze()
        twist_matrix[:, 1, 2] = -sin_theta.squeeze()
        twist_matrix[:, 2, 1] = sin_theta.squeeze()
        twist_matrix[:, 2, 2] = cos_theta.squeeze()

        deformation_matrix = torch.bmm(deformation_matrix, twist_matrix)  # Apply twist

    # 4. Tapering (Scale X and Z based on Y)
    if apply_taper:
        taper_factors = taper_factors.view(batch_size, 1)
        taper_matrix = torch.eye(3, device=device).repeat(batch_size, 1, 1)
        taper_matrix[:, 0, 1] = taper_factors.squeeze()  # Scale X based on Y
        taper_matrix[:, 2, 1] = taper_factors.squeeze()  # Scale Z based on Y

        deformation_matrix = torch.bmm(deformation_matrix, taper_matrix)  # Apply taper

    # Apply deformation matrix to the point cloud
    deformed_cloud = torch.bmm(
        point_cloud, deformation_matrix.transpose(1, 2)
    )  # Apply transformation

    # 5. Adding Noise (After transformation)
    if apply_noise:
        noise = torch.randn_like(deformed_cloud) * noise_std_factors.view(
            batch_size, 1, 1
        )
        deformed_cloud += noise

    return deformed_cloud, deformation_matrix
