# LifeAct-MDCK mechanochemical modality audit v206

## Decision

The source contains causal LifeAct state aligned to phase-contrast images for the same cells, but it does not provide ready instance masks or trajectories. The traction archive is a separate pre/post assay and cannot be used as frame-wise force supervision for these trajectories.

Therefore the defensible next experiment is a unary-state gate: reliable segmentation and identity tracking first, then LifeAct/shape/contact variables and hard temporal/wrong-cell controls. Direct force-conditioned transfer is not supported by the data contract.

## Archive inventory

| archive                      |   size_gib | remote_zip_status                  |   n_entries |   n_tiff |   n_images |   n_mask_like |   n_track_like |   n_tables |   n_xml | local_download   | checksum_ok   | url                                                                                |
|:-----------------------------|-----------:|:-----------------------------------|------------:|---------:|-----------:|--------------:|---------------:|-----------:|--------:|:-----------------|:--------------|:-----------------------------------------------------------------------------------|
| Drugs_MSD_Qt.zip             |  6.5056    | ok                                 |        1625 |     1617 |       1617 |             0 |              0 |          0 |       0 | False            |               | https://zenodo.org/api/records/20047603/files/Drugs_MSD_Qt.zip/content             |
| DynamicMCFL.zip              |  0.0316157 | ok                                 |          14 |        5 |          7 |             0 |              0 |          0 |       0 | True             | True          | https://zenodo.org/api/records/20047603/files/DynamicMCFL.zip/content              |
| Glassy_timelapse.zip         |  8.02532   | BadZipFile: File is not a zip file |           0 |        0 |          0 |             0 |              0 |          0 |       0 | False            |               | https://zenodo.org/api/records/20047603/files/Glassy_timelapse.zip/content         |
| LISA.zip                     |  8.40635   | ok                                 |         335 |      318 |        318 |             0 |              0 |          0 |       0 | False            |               | https://zenodo.org/api/records/20047603/files/LISA.zip/content                     |
| LISA_drugs_otherproteins.zip |  0.261685  | ok                                 |           6 |        5 |          5 |             0 |              0 |          0 |       0 | False            |               | https://zenodo.org/api/records/20047603/files/LISA_drugs_otherproteins.zip/content |
| LIfeact_oscillations.zip     |  3.88816   | ok                                 |         739 |      726 |        726 |             0 |              0 |          1 |       0 | False            |               | https://zenodo.org/api/records/20047603/files/LIfeact_oscillations.zip/content     |
| Traction.zip                 |  0.418742  | ok                                 |          54 |       49 |         49 |             0 |              0 |          0 |       0 | False            |               | https://zenodo.org/api/records/20047603/files/Traction.zip/content                 |

## Paired temporal candidates

| archive          | sequence_prefix                                    |   scene |   n_frames |   first_frame |   last_frame | channels   |   n_channels | z_planes   |   n_z_planes |   paired_c1_c2 |   uncompressed_gib | example                                                                                                                                       |
|:-----------------|:---------------------------------------------------|--------:|-----------:|--------------:|-------------:|:-----------|-------------:|:-----------|-------------:|---------------:|-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------|
| Drugs_MSD_Qt.zip | Drugs_MSD_Qt/Mitomycin                             |     013 |        147 |             1 |          147 | 1,2        |            2 |            |            1 |              1 |            2.29879 | Drugs_MSD_Qt/Mitomycin/s013t001c1_ORG.tif | Drugs_MSD_Qt/Mitomycin/s013t001c2_ORG.tif                                                         |
| Drugs_MSD_Qt.zip | Drugs_MSD_Qt/Y27632/s1                             |     001 |        147 |             1 |          147 | 1,2        |            2 |            |            1 |              1 |            2.29879 | Drugs_MSD_Qt/Y27632/s1/s001t001c1_ORG.tif | Drugs_MSD_Qt/Y27632/s1/s001t001c2_ORG.tif                                                         |
| LISA.zip         | LISA/LISA_timelapse/Images/LA MDCK Timelapse 49hrs |      02 |         75 |             1 |           75 | 1,2        |            2 | 1,2,3      |            3 |              1 |            2.47865 | LISA/LISA_timelapse/Images/LA MDCK Timelapse 49hrs_s02t01z1c1_ORG.tif | LISA/LISA_timelapse/Images/LA MDCK Timelapse 49hrs_s02t01z1c2_ORG.tif |

## Readiness

| candidate          | causal_lifeact_state   | aligned_segmentation_channel   | ready_masks   | ready_tracks   | framewise_force   | recommended_role                                      |
|:-------------------|:-----------------------|:-------------------------------|:--------------|:---------------|:------------------|:------------------------------------------------------|
| LISA timelapse s02 | True                   | True                           | False         | False          | False             | unary-state pilot after segmentation and tracking     |
| Mitomycin s013     | True                   | True                           | False         | False          | False             | independent-condition validation                      |
| Y27632 s001        | True                   | True                           | False         | False          | False             | independent-condition validation                      |
| Traction archive   | False                  | False                          | False         | False          | False             | mechanistic interpretation only; not a per-step input |

## Integrity notes

- Glassy_timelapse.zip: BadZipFile: File is not a zip file
