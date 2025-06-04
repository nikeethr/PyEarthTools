set unstable := true

# ==============================================================================
#  STEPS:
# ==============================================================================
#
# +----------------------------------------------+
# ! 1. launch_nix_shell_interactive -> important !
# +----------------------------------------------+
#   2. create_workpaths
#   3. copy_repo_to_workdir
#   4. cd to repo workdir (manual)
#   5. pixi init
#   6. start developing
#   *. add more convenient tasks, adjust nix definition, custom jupyter kernel etc.
#
# The above could be done in a non-interactive shell but I'm too lazy for now.
#
# ==============================================================================
#  variables for justfile
# ==============================================================================
# project settings - config files etc. read-only

projecthome := "/g/data/kd24/nr4547"
projectdir := projecthome / "work/dev"
projectgit := "git@github.com:ACCESS-Community-Hub/PyEarthTools.git"
nixconf := projectdir / "nix.conf"
nixstatic := projectdir / "nixstatic"
projectname := "PyEarthTools"
nixshelldef := justfile_dir() / "shell.nix"
nixchannel := "https://github.com/NixOS/nixpkgs/archive/nixos-24.11.tar.gz"

# work settings - temporary directory etc. to do stuff in

workdir := `realpath "${PBS_JOBFS:-/var/tmp}/nix-nr4547"`
repo_workdir := workdir / projectname
nixpath_workdir := workdir / "nixpkgs"
pixihome := workdir / ".pixi"

# ==============================================================================
#  env variables
# ==============================================================================
#
# set paths

export PATH := `echo ${PATH}:` + nixstatic + "/bin:" + nixstatic + "/lib"

# set temporary work dir paths

export TMPDIR := workdir / "tmp"
export XDG_CONFIG_HOME := workdir / ".config"
export XDG_CACHE_HOME := workdir / ".cache"
export XDG_DATA_HOME := workdir / ".local" / "share"
export XDG_STATE_HOME := workdir / ".local" / "state"
export JUPYTER_DATA_DIR := env('HOME') / ".local/share/jupyter"

# nix environment variables

export NIX_SSL_CERT_FILE := "/half-root/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"
export NIX_USER_CONF_FILES := nixconf
export NIX_PATH := "nixpkgs=" + nixpath_workdir

# repeat for singularity

export SINGULARITYENV_TMPDIR := TMPDIR
export SINGULARITYENV_XDG_CONFIG_HOME := XDG_CONFIG_HOME
export SINGULARITYENV_XDG_CACHE_HOME := XDG_CACHE_HOME
export SINGULARITYENV_XDG_DATA_HOME := XDG_DATA_HOME
export SINGULARITYENV_XDG_STATE_HOME := XDG_STATE_HOME
export SINGULARITYENV_NIX_SSL_CERT_FILE := NIX_SSL_CERT_FILE
export SINGULARITYENV_NIX_USER_CONF_FILES := nixconf
export SINGULARITYENV_NIX_PATH := NIX_PATH
export SINGULARITYENV_PREPEND_PATH := nixstatic / "bin"
export SINGULARITYENV_PIXI_HOME := pixihome
export SINGULARITYENV_PBS_JOBFS := `echo ${PBS_JOBFS:-/var/tmp}`
export SINGULARITYENV_JUPYTER_DATA_DIR := JUPYTER_DATA_DIR

# command strings

singularity_exec := '''
singularity exec \
    --cleanenv \
    --bind "''' + XDG_DATA_HOME + '''/nix/root/nix":/nix \
    --bind "''' + NIX_SSL_CERT_FILE + '''" \
    --bind "''' + projecthome + '''" \
    --bind "''' + workdir + '''" \
    --bind /half-root/usr/lib64/:/blah/lib64 \
    --writable-tmpfs \
    --env-file "''' + justfile_dir() + '''/singularity.env" \
    "''' + projecthome  + '''/work/dev/debian_latest.sif" \
'''
kernel_script := '''
#!/bin/bash
__connection_str=$1

cd ''' + justfile_dir() + '''

module load singularity
''' + singularity_exec + ''' nix-shell shell.nix \
    --run "pixi run -e default python -m ipykernel_launcher -f ${__connection_str}"
'''
jupyter_kernel := '''
{
 "argv": [
   "''' + justfile_dir() + '''/launch_kernel.sh",
   "{connection_file}"
 ],
 "display_name": "Python 3 (PET-pixienv=all)",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
'''

# ==============================================================================
# recipes
# ==============================================================================
#
# ------------------------------------------------------------------------------
# these happen in gadi host shell

alias nscl := launch_nixshell_clean
alias nscx := launch_nixshell_cache
alias nsxx := launch_nixshell

make_singularity_envfile:
    #!/bin/bash
    __envfile="{{ justfile_dir() }}/singularity.env"
    echo "TMPDIR={{ SINGULARITYENV_TMPDIR }}" > "$__envfile"
    echo "XDG_CONFIG_HOME={{ SINGULARITYENV_XDG_CONFIG_HOME }}" >> "$__envfile"
    echo "XDG_CACHE_HOME={{ SINGULARITYENV_XDG_CACHE_HOME }}" >> "$__envfile"
    echo "XDG_DATA_HOME={{ SINGULARITYENV_XDG_DATA_HOME }}" >> "$__envfile"
    echo "XDG_STATE_HOME={{ SINGULARITYENV_XDG_STATE_HOME }}" >> "$__envfile"
    echo "NIX_SSL_CERT_FILE={{ SINGULARITYENV_NIX_SSL_CERT_FILE }}" >> "$__envfile"
    echo "NIX_USER_CONF_FILES={{ SINGULARITYENV_NIX_USER_CONF_FILES }}" >> "$__envfile"
    echo "NIX_PATH='{{ SINGULARITYENV_NIX_PATH }}'" >> "$__envfile"
    echo "PREPEND_PATH={{ SINGULARITYENV_PREPEND_PATH }}" >> "$__envfile"
    echo "PIXI_HOME={{ SINGULARITYENV_PIXI_HOME }}" >> "$__envfile"
    echo "PBS_JOBFS={{ SINGULARITYENV_PBS_JOBFS }}" >> "$__envfile"
    echo "JUPYTER_DATA_DIR={{ SINGULARITYENV_JUPYTER_DATA_DIR }}" >> "$__envfile"

clean_workdir:
    #!/bin/bash
    set -eu -o pipefail
    chmod +rwX -R "{{ workdir }}" || echo "{{ workdir }} not found"
    rm -rf  "{{ workdir }}" || echo "{{ workdir }} not found"

create_workpaths:
    #!/bin/bash
    set -eu -o pipefail
    mkdir -p "$TMPDIR"
    mkdir -p "$XDG_CONFIG_HOME"
    mkdir -p "$XDG_CACHE_HOME"
    mkdir -p "$XDG_DATA_HOME"
    mkdir -p "$XDG_STATE_HOME"

download_nixpkgs:
    #!/bin/bash
    set -eu -o pipefail
    cd $TMPDIR
    rm -rf nixpkgs*
    curl -L --tlsv1.2 -o "nixpkgs.tar.gz" -- "{{ nixchannel }}"
    tar xzf "nixpkgs.tar.gz" && rm -rf "nixpkgs.tar.gz"
    mv nixpkgs* "{{ nixpath_workdir }}"

launch_nixshell: make_singularity_envfile
    #!/bin/bash -i
    module load singularity
    mkdir -p "{{ workdir }}/nix"
    echo $PATH
    # psuedo build bash to initialize necessary directories
    nix-build \
        -I "nixpkgs={{ nixpath_workdir }}" '<nixpkgs>' \
        --attr bashInteractive \
        --no-out-link
    # actual execution in container to mock /nix/store as chroot store
    {{ singularity_exec }} nix-shell -I "nixpkgs={{ nixpath_workdir }}" shell.nix

unsquash_work target: clean_workdir
    #!/usr/bin/env bash
    set -eu -o pipefail
    module load singularity  # might not be required
    unsquashfs -d "{{ workdir }}" -f "{{ target }}"

unsquash_latest_work: clean_workdir
    #!/usr/bin/env bash
    set -eu -o pipefail
    module load singularity  # might not be required
    unsquashfs -d "{{ workdir }}" -f "{{ projectdir }}/pet_latest.squashfs"

pixi_init_noclean:
    #!/bin/bash
    # only relink symlink
    rm -r .pixi
    ln -s "{{ workdir }}/pet/.pixi" .pixi
    cp "{{ repo_workdir }}/pyproject.toml" pyproject.toml 
    cp "{{ repo_workdir }}/pixi.lock" pixi.lock
    # pixi init needs to be done manually

install_kernel_spec: make_singularity_envfile
    #!/bin/bash
    set -eu -o pipefail
    # redirect to local jupyter lab
    __kernel_dir="${JUPYTER_DATA_DIR}/kernels/pet-all"
    # launch kernel script
    echo '{{ kernel_script }}' > "{{ justfile_dir() }}/launch_kernel.sh"
    chmod +x "{{ justfile_dir() }}/launch_kernel.sh"
    # kernel spec ... uses launch kernel script
    mkdir -p "${__kernel_dir}"
    echo '{{ jupyter_kernel }}' > "${__kernel_dir}/kernel.json"

[doc("launch shell from scratch, cleaning and downloading everything - slow, but reproducible")]
launch_nixshell_clean: clean_workdir create_workpaths download_nixpkgs install_kernel_spec pixi_init_clean launch_nixshell

[doc("launch from existing squashed (cached) workdir - fast, but may have outdated stuff")]
launch_nixshell_cache: unsquash_latest_work install_kernel_spec pixi_init_noclean launch_nixshell

# ------------------------------------------------------------------------------
# These happen in nix-shell...

clone_repo_to_workdir:
    #!/usr/bin/env bash
    set -eu -o pipefail
    mkdir -p $(dirname "{{ repo_workdir }}")
    git clone "{{ projectgit }}" "{{ repo_workdir }}" || true

pixi_init_clean:
    #!/usr/bin/env bash
    pixi clean
    # handle global environments
    mkdir -p $PIXI_HOME
    # symlink to workdir
    rm -r .pixi
    ln -s "{{ workdir }}/pet/.pixi" .pixi
    cp "{{ repo_workdir }}/pyproject.toml" pyproject.toml 
    cp "{{ repo_workdir }}/pixi.lock" pixi.lock
    pixi init

pixi_add_jup:
    #!/usr/bin/env bash
    pixi add jupyter    # jupyterlab
    pixi add ipykernel  # ipykernel to connect to jupyter lab

jup target="all":
    #!/usr/bin/env bash
    pixi run -e "{{ target }}" -- jupyter lab --no-browser

squash_work:
    #!/usr/bin/env bash
    set -eu -o pipefail
    __dest="{{ projectdir }}/pet_$(date +%Y%m%d_%H%M%S).squashfs"
    cp pyproject.toml "{{ repo_workdir }}/pyproject.toml"
    cp pixi.lock "{{ repo_workdir }}/pixi.lock"
    mksquashfs "{{ workdir }}" "${__dest}"
    rm -f "{{ projectdir }}/pet_latest.squashfs" || true
    ln -s "${__dest}" "{{ projectdir }}/pet_latest.squashfs"
