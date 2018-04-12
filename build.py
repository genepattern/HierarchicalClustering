from subprocess import call
call("mkdir -p archive/current/", shell=True)

# Turn these into an input
module_name = "HierarchicalClustering"
path_to_manifest = "manifest"

# reading the old version:
def get_version(line):
    idx = line[::-1].find(":")  # Find the last occurrence of ":"
    return line[-idx:].strip("\n"), idx  # Return all the values after the last occurrence of ":"

def read_manifest_version(file_name):
    manifest = open(file_name)
    siguiente = True
    version = '99.99'
    while siguiente:
        line = manifest.readline()
        if line[:4] == "LSID":
            version, _ = get_version(line)
            siguiente = False
    return version

old_version = read_manifest_version(path_to_manifest)
# For now update only the minor version
version_vector = old_version.split(".")
version_vector[1] = str(int(version_vector[1])+1)
current_version = ".".join(version_vector)
print("Old version is", old_version)
print("Updated version is", current_version)


# update the manifest version
def update_manifest_version(file_name,old_version, new_version):
    call(" ".join(["cp", file_name, "temp"]), shell=True)
    old_manifest = open("temp", 'r')
    new_manifest = open(file_name, 'w')
    for line in old_manifest.readlines():
        if line[:4] == "LSID":
            _, idx = get_version(line)
            # line = line[:idx] + new_version
            line = line.replace(old_version+"\n", new_version+"\n")
        new_manifest.write(line)
    new_manifest.close()
    return

update_manifest_version(path_to_manifest, old_version, current_version)

path_to_archive = "archive/"+module_name+"."+current_version+".zip"
path_to_current = "archive/current/"+module_name+"."+current_version+".zip"

# path_to_src = "src"
what_to_zip = "src/manifest src/HierarchicalClustering.py src/hc_functions.py src/test_HC.py src/command_line.py " \
              "src/doc.html"

# Saving to the archive:
print("Zipping to archive")
# call(["cd "+path_to_src, "ls", " ".join(["zip", ".."+path_to_archive, what_to_zip])], shell=True)
call(" ".join(["zip -j", path_to_archive, what_to_zip]), shell=True)
# call("cd ..", shell=True)

# Saving to the current:
call("rm -r archive/current/", shell=True)
call("mkdir archive/current/", shell=True)
call(" ".join(["cp", path_to_archive, path_to_current]), shell=True)


'''
Probably need to call something like this:

import os
script = """
echo $0
ls -l
echo done
"""
os.system("bash -c '%s'" % script)

OOOOOR

import subprocess
script = """
echo $0
ls -l
echo done
"""
do a subprocess.call(script, shell=True)


Yeah, that's better.
'''
