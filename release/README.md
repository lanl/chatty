# Laptop suite releases

Chatty hosts the three application wheels, two pip requirements files, the JSON
manifest and runner evidence together. A suite tag such as `laptop-2026.11.1`
identifies one set of bytes independently of the three Python package versions.
The tag points at the Chatty application revision in the manifest. Each release's
notes link the revision of the release tooling and the build and verification runs.

## Reproduce a published release

Use the copy of this directory linked in the release notes. The SHA-256 in
`manifest.sha256` anchors the manifest, which in turn identifies the source commit,
wheel URL and digest of every application, and the exact wheel of every runtime
dependency on each platform. The README links to self-contained installers at a
specific Git commit. Each installer verifies its uv archive, the suite manifest
and its platform requirements file before invoking pip. `--no-index` and explicit wheel URLs select
the intended projects even when the same names exist on PyPI. The suite requests
`chatty[rag]` and supplies the LANL LitKit wheel explicitly. The source-development
`litkit` extra uses Git and is outside this install path.

To rebuild the application wheels on Ubuntu 24.04, start in the Chatty repository
at the release-tooling revision. Use the CPython patch version in `.python-version`
and a fresh build environment. For example, for the first suite:

```bash
uv venv --managed-python --python 3.12.12 /tmp/chatty-wheel-build
uv pip install --python /tmp/chatty-wheel-build --only-binary=:all: -r release/build-requirements.txt
gh release download laptop-2026.11.1 -R lanl/chatty -p manifest.json -D dist
/tmp/chatty-wheel-build/bin/python release/prepare.py reproduce dist
```

The builder checks out each full Git SHA in a fresh directory, uses the commit's
timestamp for `SOURCE_DATE_EPOCH`, and builds with pinned backends and disabled
build isolation. Installing a released set only downloads wheels. Resolving against
today's package index is only part of cutting a **new** set.

## Cut the next suite

1. Read each project's contributor and release instructions. Select the source
   revisions and verify that each wheel's metadata version agrees with its CLI.
   Update the three versions and full SHAs in `sources.json`. Allocate a new
   `laptop-YYYY.MM.N` tag and update the manifest link in the top-level README.
   If Python changes, update `.python-version` and the README version. Review the starter constraints against the upstream locks,
   especially the Windows FAISS and CPU Torch requirements.
2. Run **Prepare laptop release** on the release branch. It runs on PRs that
   change the build inputs and can also be dispatched after the workflow lands
   on main. Linux builds the wheels once. The two native runners resolve from
   those wheels with binary-only pip. The assemble job checks both reports
   against the built wheels and emits `laptop-release`.
3. Download that artifact to a new directory. Record the manifest digest in
   the release PR. Stage the six files together as a public GitHub prerelease
   so the native verification jobs can use the advertised URLs. The following
   commands use Bash with `prepare_run` set to the successful Actions run ID:

   ```bash
   gh run download "$prepare_run" -R lanl/chatty -n laptop-release -D candidate
   (cd candidate && shasum -a 256 manifest.json) > release/manifest.sha256
   tag=$(jq -r .tag candidate/manifest.json)
   revision=$(jq -r '.projects.chatty.revision' candidate/manifest.json)
   gh release create "$tag" candidate/*.whl candidate/*.txt candidate/manifest.json \
     -R lanl/chatty --target "$revision" --prerelease --title "Laptop suite $tag" \
     --notes "Candidate for the Python 3.12 laptop suite. Build run $prepare_run."
   ```

4. Set the suite tag and manifest SHA-256 in `install.sh` and `install.ps1`.
   If changing uv, update its version and archive SHA-256 for each platform.
   Commit the installers and `manifest.sha256` with `git commit -s` and push.
   In a second commit, pin both marked README commands to that installer commit's
   full SHA. This keeps each release installable after the next upgrade.
   **Verify published laptop wheels** runs the literal commands from outside a
   checkout as standard users on both targets. Only OS tools are on the install
   PATH. The verifier checks the published installer bytes, all reported wheels,
   fresh-shell command discovery, the managed Python and Chatty's LitKit imports.
   It repeats the install and checks that the corpus, FAISS index, user config
   and saved Chatty session survive a runtime switch. The first release uses a
   simulated previous release marker. Keep its installer revision as the
   baseline for the first upgrade between published suites.
   The Linux job reproduces all three wheel hashes. All three checks must pass.
   If a wheel or lock needs to change, allocate a new suite tag.
5. Download the two evidence files, inspect their CLI versions and module paths,
   and attach them to the same release. Put the manifest digest, preparation and
   verification run URLs, tooling commit and companion PRs in the release notes.
   After review, promote the verified release in place. For example, with
   `verify_run` and `tag` set:

   ```bash
   gh run download "$verify_run" -R lanl/chatty -n evidence-macos-arm64 -D evidence
   gh run download "$verify_run" -R lanl/chatty -n evidence-windows-x64 -D evidence
   gh release upload "$tag" evidence/*.evidence.json -R lanl/chatty
   gh release edit "$tag" -R lanl/chatty --prerelease=false --notes-file release-notes.md
   ```

The evidence JSON files remain available as release assets after Actions
artifact retention expires. The wheel and lock URLs stay the same on promotion.
