# vllm-rdna-docker Configuration Schema

**Minimum Python: 3.11** (uses stdlib `tomllib`; no third-party TOML package required).

This document is the contract for every `vllm-rdna-docker/config/*.toml` file.
It is enforced by `vllm-rdna-docker/tools/validate.py`. Anything the validator
accepts must be documented here; anything documented here must be enforced by
the validator. Later todos (resolver, build CLI, publish, CI) consume only
configs that pass this schema.

## Design invariants

1. **Fixed architecture set.** `[architecture].targets` must contain exactly
   the seven RDNA targets, and nothing else:
   `gfx1030, gfx1100, gfx1101, gfx1150, gfx1151, gfx1200, gfx1201`.
   There is **no `primary` architecture concept**: the key `primary`
   (any case) is forbidden at **every nesting level** of the document.
2. **Explicit matrix, no Cartesian product.** Images are declared one by one.
   A source additionally declares `compatible_bases`, and an image combining a
   base outside that list is rejected as an unapproved combination before any
   build command is ever emitted.
3. **Immutability.** Sources pin a full 40-character commit; bases pin a full
   `sha256:` digest; artifacts pin a `sha256` checksum. Tags listed in
   `[project].reserved_tags` (legacy `v0.22.1`, `v0.22.1_base`) may never be
   claimed by a new image or alias.
4. **No host state.** No absolute host paths, no ambient checkouts, no
   build-engine assumptions. The build machine is external and Podman-based;
   nothing in a config may reference `/home/...` or similar paths.
5. **No side effects from validation.** The validator is read-only: it never
   writes files, never resolves refs over the network, never touches a
   container engine.

## Top-level sections

| Section      | Required | Purpose |
|--------------|----------|---------|
| `[project]`    | yes | Project metadata, schema version, reserved legacy tags |
| `[architecture]` | yes | The fixed seven-target architecture set |
| `[registry]`   | yes | Registry host and image repository names |
| `[bases.<key>]`  | yes (>=1) | ROCm base stack definitions |
| `[sources.<key>]` | yes (>=1) | vLLM source definitions (upstream releases, extras-fork versions) |
| `[[images]]`   | yes (>=1) | Explicit approved base x source image combinations |
| `[aliases]`    | no | Unqualified tag -> image id promotion pointers |
| `[patches.<key>]` | no (EXPERIMENTAL) | Checksummed patch entries referenced by bases |
| `[artifacts.<key>]` | no (EXPERIMENTAL) | External artifacts with checksums |

Any other top-level section is rejected (`UnknownSection`). Unknown keys
inside a known section are rejected (`UnknownField`) — the schema is strict,
not permissive; extend it deliberately.

### Key vs `id` convention (bases, sources, patches, artifacts)

The TOML table key (`[bases.rocm720]` -> key `rocm720`) is a human-readable
label. The **required `id` field inside the table is the canonical identity**
referenced everywhere else (`base = "rocm720"`, `compatible_bases = [...]`).
Keeping key and id aligned is recommended. The validator rejects two entries
declaring the same `id` (`DuplicateSourceId`, `DuplicateBaseId`, ...) — this
catches accidental duplicates that TOML's own duplicate-key rule cannot,
because TOML only forbids identical *keys*, not identical *identities*.

## `[project]` (required)

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `name` | string | yes | non-empty |
| `schema_version` | integer | yes | must equal `1` |
| `description` | string | no | non-empty if present |
| `reserved_tags` | array<string> | no | tags no image or alias may claim (legacy images live here) |

## `[architecture]` (required)

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `targets` | array<string> | yes | exactly the seven required targets, unique, no extras |

## `[registry]` (required)

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `host` | string | yes | e.g. `docker.io` |
| `base_repository` | string | yes | e.g. `blivioniag/rocm-rdna` |
| `vllm_repository` | string | yes | e.g. `blivioniag/vllm-rdna` |

## `[bases.<key>]` (at least one required)

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `id` | string | yes | unique across bases |
| `rocm_version` | string | yes | e.g. `7.2.0` |
| `python_version` | string | yes | e.g. `3.12` |
| `pytorch_version` | string | yes | e.g. `2.12.0` |
| `triton_version` | string | yes | e.g. `3.6.0` |
| `base_image` | string | yes | upstream image ref, e.g. `rocm/dev-ubuntu-22.04:7.2.0-complete` |
| `base_digest` | string | yes | `sha256:` + 64 lowercase hex |
| `pytorch_index_url` | string | yes | wheel index for the torch stack, e.g. `https://download.pytorch.org/whl/rocm7.2`; per base, no project default |
| `tag` | string | yes | published tag under `registry.base_repository` |
| `patches` | array<string> | no | each entry must be a defined patch id (`UnknownPatch`) |
| `flash_attention` | table | no | optional `[bases.<key>.flash_attention]` sub-table, see below |

## `[bases.<key>.flash_attention]` (optional)

Controls whether and where Flash Attention is installed for this base's
images. Absent means not installed (equivalent to `install = "none"`).

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `install` | string | yes | one of `base` (install into the base image), `vllm` (install into the application image), `none` |
| `version` | string | when `install != "none"` | FA version pin, e.g. `2.7.4` |
| `repo` | string | no | clone URL, default `https://github.com/Dao-AILab/flash-attention`; no host paths |
| `ref` | string | no | branch/tag/commit to clone, default `main` |

### Per-base upstream image

`base_image` is required **per base** — there is NO project-level default
image. Each ROCm version pins its own concrete upstream image reference and
its own immutable `base_digest` (for example
`rocm/dev-ubuntu-22.04:7.2-complete` for ROCm 7.2.0 and
`rocm/dev-ubuntu-22.04:7.14.0-full` for ROCm 7.14.0). Two bases may never
implicitly share an image: if two ROCm versions happen to use the same
upstream tag, that choice must be written out explicitly in both entries, and
each entry still carries its own digest. Host paths are not allowed in
`base_image` (`HostPathNotAllowed`).

## `[sources.<key>]` (at least one required)

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `id` | string | yes | unique across sources |
| `variant` | string | yes | one of `upstream`, `extras-fork` |
| `repository` | string | yes | clone URL; must not be a host path |
| `version` | string | yes | upstream compatibility version, e.g. `0.26.0` |
| `ref` | string | yes | git ref (tag/branch), e.g. `v0.26.0` |
| `commit` | string | yes | full 40-char lowercase hex, must be consistent with `ref` (verified by the resolver in Todo 2, not here) |
| `compatible_bases` | array<string> | yes | every entry must be a defined base id (`UnknownBase`); non-empty |

## `[[images]]` (at least one required)

Each entry is one explicit approved combination.

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `id` | string | yes | unique across images (`DuplicateImageId`) |
| `base` | string | yes | must be a defined base id (`UnknownBase`) |
| `source` | string | yes | must be a defined source id (`UnknownSource`) |
| `tag` | string | yes | unique across images (`DuplicateTag`); not in `reserved_tags` (`ReservedTag`) |

Approval rule: `image.base` must appear in `source.compatible_bases`,
otherwise `UnapprovedCombination(base=..., source=...)`.

## `[aliases]` (optional)

Unqualified tag -> image id. `v0.26.0 = "vllm-026-upstream-rocm720"` declares
that the unqualified tag `v0.26.0` resolves to that image. Rules:

- the target image id must exist (`UnknownImage`);
- the alias name must equal the target image's `tag` (`AliasTagMismatch`) —
  an alias is the *promotion pointer for the tag the image itself produces*,
  never an arbitrary nickname;
- the alias must not be in `reserved_tags` (`ReservedTag`).

Qualified tags (e.g. `v0.26.0-rocm7.14.0`) are produced by their images
directly and do not need aliases.

## `[patches.<key>]` / `[artifacts.<key>]` (optional, EXPERIMENTAL)

Provisional shape; Todo 3+ may extend. Current rules:

- patches: `id` (unique), `sha256` (64 lowercase hex), optional `description`, optional `base` (must be a defined base id if present);
- artifacts: `id` (unique), `url` (non-empty, not a host path), `sha256`.

## Named errors

Every validation failure raises a subclass of `ConfigError`. `str(err)` renders
`Name(field=value, ...)`; `err.context` holds the structured fields.

| Error | Context | Raised when |
|-------|---------|-------------|
| `MalformedTOML` | `detail` | file does not parse as TOML |
| `UnknownSection` | `section` | top-level section not in the allowlist |
| `MissingSection` | `section` | required top-level section absent |
| `EmptySection` | `section` | `bases`/`sources`/`images` present but empty (at least one entry required) |
| `UnknownField` | `section`, `field` | key not allowed inside a known section |
| `MissingField` | `section`, `field` | required field absent or empty |
| `InvalidFieldType` | `section`, `field`, `expected` | field has the wrong TOML type |
| `InvalidSchemaVersion` | `found` | `project.schema_version != 1` |
| `PrimaryArchitectureNotAllowed` | `path` | a key named `primary` (any case) found at any depth |
| `MissingArchitecture` | `arch` | a required target is absent from `architecture.targets` |
| `UnexpectedArchitecture` | `arch` | a target outside the fixed seven is present |
| `DuplicateArchitecture` | `arch` | a target listed twice |
| `DuplicateBaseId` | `base` | two base entries declare the same `id` |
| `DuplicateSourceId` | `source` | two source entries declare the same `id` |
| `DuplicateImageId` | `image` | two images declare the same `id` |
| `DuplicatePatchId` | `patch` | two patch entries declare the same `id` |
| `DuplicateArtifactId` | `artifact` | two artifact entries declare the same `id` |
| `UnknownBase` | `base` | image/patch/compatible_bases references an undefined base |
| `UnknownSource` | `source` | image references an undefined source |
| `UnknownImage` | `image` | alias targets an undefined image |
| `UnknownPatch` | `patch` | base references an undefined patch |
| `InvalidVariant` | `source`, `variant` | source variant not in {`upstream`, `extras-fork`} |
| `InvalidFlashAttentionInstall` | `base`, `install` | flash_attention install not in {`base`, `vllm`, `none`} |
| `InvalidCommit` | `source`, `commit` | commit is not 40 lowercase hex chars |
| `InvalidDigest` | `base`, `digest` | base_digest is not `sha256:` + 64 lowercase hex |
| `InvalidChecksum` | `section`, `field` | sha256 field is not 64 lowercase hex |
| `UnapprovedCombination` | `base`, `source` | image pairs a base not in the source's `compatible_bases` |
| `DuplicateTag` | `tag` | two images produce the same tag |
| `ReservedTag` | `tag` | image tag or alias claims a reserved legacy tag |
| `AliasTagMismatch` | `alias`, `image` | alias name does not equal the target image's tag |
| `HostPathNotAllowed` | `section`, `field`, `value` | field embeds an absolute host path (`/home/...`, `/Users/...`, `/root/...`) |

## Extensibility rules for Todos 2-8

- New fields must be added here **and** to the validator in the same change;
  unknown fields are rejected, so an undocumented field cannot silently land.
- New top-level sections extend the allowlist deliberately.
- `[patches]` / `[artifacts]` are marked EXPERIMENTAL: their shape may grow
  (per-base patch lists, artifact extraction hints) without a schema-version
  bump, but existing required fields may not change meaning.
- `schema_version` stays `1` until a breaking change; a bump requires a
  migration note here.
