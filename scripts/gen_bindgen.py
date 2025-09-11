# # Bindgen of TBLIS (tblis.h)

import subprocess
import os
import shutil
import re
from tree_sitter import Language, Parser
import tree_sitter_rust

import util_dyload

path_cwd = os.path.abspath(os.getcwd())

# ## Bindgen configuration

# Users may change the following fields for their needs.

# Path for storing useful header files
path_header = f"{path_cwd}/../header"

# Path for temporary files
path_temp = f"{path_cwd}/tmp"

# Path for bindgen crate root
path_out = f"{path_cwd}/../tblis-ffi"

# ## Copy necessary headers

shutil.copytree(path_header, path_temp, dirs_exist_ok=True)

# +
# From now on, we will always work in temporary directory

os.chdir(path_temp)
# -

# ## Pre-processing headers

# We will disable explicit definition of tci.

# +
with open("tblis/frame/base/thread.h", "r") as f:
    token = f.read()

token = token \
    .replace("typedef tci_comm", "typedef void") \
    .replace('#include "tci.h"', "") \
    .replace('#include "tci/mutex.h"', "") \

with open("tblis/frame/base/thread.h", "w") as f:
    f.write(token)
# -

# ## Bindgen generation

subprocess.run([
    "bindgen",
    "tblis.h", "-o", "tblis.rs",
    "--allowlist-file", "./tblis/frame/[\\S]*.h",
    "--default-enum-style", "rust",
    "--no-layout-tests",
    "--use-core",
    "--merge-extern-blocks",
    "--", "-I.",
])

# ## Post-processing

with open("tblis.rs", "r") as f:
    token = f.read()

token = """

use num::Complex;
use core::ffi::*;

""" + token

# +
# use num::Complex instead of bindgen complex

token = re.sub(r"\#\[derive[^=]*__BindgenComplex<T> {[^#]*?}", "", token)
token = token.replace("__BindgenComplex", "Complex")
# -

token = token.replace("::core::ffi::", "")

# ## Dynamic loading split

files_split = util_dyload.dyload_main(token)

# +
dir_relative = "tblis"

shutil.rmtree(dir_relative, ignore_errors=True)
os.makedirs(dir_relative)
for key, string in [
    ("ffi_base", files_split["ffi_base"]),
    ("ffi_extern", files_split["ffi_extern"]),
    ("dyload_initializer", files_split["dyload_initializer"]),
    ("dyload_struct", files_split["dyload_struct"]),
    ("dyload_compatible", files_split["dyload_compatible"]),
    ("mod_template", files_split["mod_template"])
]:
    with open(f"{dir_relative}/{key}.rs", "w") as f:
        f.write(string)
# -

shutil.copytree(f"{path_temp}/{dir_relative}", f"{path_out}/src/{dir_relative}", dirs_exist_ok=True)
