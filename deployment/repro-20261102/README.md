# GLM5.3 shared project snapshot — 2026-11-02

This is a preservation snapshot of the current working source, not a refactor or a new training validation. The five private repositories retain their source ancestry. The source working directories and their Git metadata were not edited. Existing missing tracked files are preserved as deletions in the source commits; they are not claimed as functional fixes.

## Branches and commits

Use `snapshot/glm53-shared-20261102` in each private repository. `repositories.json` records the exact four dependency snapshot commits and the initial outer repository source commit; later commits preserve existing Git entries, update its Slime gitlink and recovery metadata. The outer `.gitmodules` points to the private Slime repository and the gitlink pins its exact snapshot commit.

`source/*-20261102` branches preserve refs observed in the working repository. `source/base-20261102` preserves its original HEAD. `upstream/base-*` identifies a verified upstream baseline where available. A detached HEAD is recorded as detached; it is not relabeled as upstream main. These are the source refs observed at capture time, not a mirror of every current upstream branch. The existing Slime private main and prior snapshot branches are retained.

## Restore source layout

On a replacement machine or an empty recovery workspace, clone this outer repository into `slime-ascend`, using the snapshot branch, then run `git submodule update --init --recursive`. Clone the other three private dependency repositories into the exact project-relative paths in `repositories.json` and check out their recorded commit IDs. Preserve the original capitalization and `sgl-kernel-npu-src` directory name.

`dependencies.json` pins remaining source repositories to their actual origins and commits. Several come from specific forks or development branches: Megatron-Bridge is from fzyzcjy, Triton Ascend kernels uses the captured shenzhaofeng KDA branch, and FLA-NPU uses the captured v26.10.0 source. Do not replace them with a similarly named upstream main branch. No new private repositories are created for these dependencies. Their pre-existing deletion-only states are recorded for provenance, and the complete earlier project archive remains available if that exact filesystem state is required.

Copy `project-files/activate.sh`, `project-files/README.md`, `project-files/AGENTS.md`, and the `project-files/start/` scripts back to the project root with their modes preserved. These are byte-for-byte copies; the live `start/` directory was not edited or moved. Logs and profiling outputs are excluded from Git. Original launchers contain absolute paths and relative layout assumptions: exact recovery uses `/mnt/share/l00804615/glm5.3_flash_native` on a replacement machine. Never overlay recovery files onto an active checkout.

## Installed environment and external assets

Git alone is not the installed runtime. Environment inventories were captured earlier on the same day in the backup directory named in `repositories.json`. `installed-versions.txt` is an inventory, not a pip lockfile or a complete reinstall recipe. Original wheels are missing for some installed packages. Keep the installed Conda and CANN archives listed in `archive-checksums.json` on durable storage; their absolute path in Git is only a reference, not an uploaded backup.

Verify SHA-256 before restoring archives. Restore the complete installed Conda environment to `/mnt/share/l00804615/conda/envs/lhw_glm_53_flash_native` and CANN to `/mnt/share/l00804615/cann/9.2.0`, on a matching aarch64 Ascend host with compatible OS, driver and firmware. `/opt/miniconda3`, mounts, networking and system libraries remain prerequisites. Do not inject task-private binaries from cache. Future native changes must use the normal source/build/package/install workflow.

`dependency-roles.json` records launcher PYTHONPATH precedence and editable versus installed dependencies. Source and installed Megatron optimizer files differ; the launcher uses the project source. Installed native binary equivalence to every source checkout is not established. The ops-transformer header-path adjustment is intentionally not submitted. The environment archive preserves the existing installed runtime without claiming it can be rebuilt from unmodified upstream sources alone.

Weights and datasets listed in `external-assets.json` are external and were not uploaded. Large weight shards have size/mtime records, not full content hashes. Retain the exact assets separately. The prior `project-recovery.tar.zst` is an earlier whole-project backup; it is not a substitute for these newly committed source snapshots if concurrent source work changed since that backup.

## Validation boundary

The selected source files were hash-compared with the live working trees; Git metadata was checked unchanged. This task does not build, install, run training, or claim a clean-machine reproduction test. Import, device, numerical, performance and long-run checks on a replacement machine remain unperformed. Historical deployment/environment files in the inner Slime repository are retained unchanged; use this dated outer-project manifest for this snapshot.
