#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
ModelPath="${ScriptPath}/../model"
BuildPath="${ScriptPath}/../build/intermediates/host"

function build()
{
  # Ensure current working directory is valid before cleaning build outputs.
  cd "${ScriptPath}" || return 1

  if [ -d "${BuildPath}" ];then
    rm -rf "${BuildPath}"
  fi

  mkdir -p "${BuildPath}"
  cd "${BuildPath}" || return 1

  local c_compiler="${CC}"
  local cxx_compiler="${CXX}"
  if [ -z "${c_compiler}" ]; then
    c_compiler=$(command -v gcc)
  fi
  if [ -z "${cxx_compiler}" ]; then
    cxx_compiler=$(command -v g++)
  fi

  if [ -z "${c_compiler}" ] || [ -z "${cxx_compiler}" ]; then
    echo "[ERROR] gcc/g++ not found. Please install compiler or set CC/CXX."
    return 1
  fi

  echo "[INFO] Using C compiler: ${c_compiler}"
  echo "[INFO] Using CXX compiler: ${cxx_compiler}"

  cmake ../../../ -DCMAKE_C_COMPILER="${c_compiler}" -DCMAKE_CXX_COMPILER="${cxx_compiler}"
  if [ $? -ne 0 ];then
    echo "[ERROR] cmake error, Please check your environment!"
    return 1
  fi
  make
  if [ $? -ne 0 ];then
    echo "[ERROR] build failed, Please check your environment!"
    return 1
  fi

  cd "${ScriptPath}" || return 1
}
function main()
{
  echo "[INFO] Sample preparation"

 ret=`find ${ModelPath} -maxdepth 1 -name yolov8s.om 2> /dev/null`
 static_ret=`find ${ModelPath} -maxdepth 1 -name yolov8s_static.om 2> /dev/null`

   if [[ ${ret} && ${static_ret} ]];then
      echo "[INFO] The yolov8s.om and yolov8s_static.om already exist, start building"
    else
      echo "[ERROR] yolov8s.om or yolov8s_static.om does not exist, please place them under ${ModelPath}"
      return 1
    fi

    
  build
  if [ $? -ne 0 ];then
    return 1
  fi
    
  echo "[INFO] Sample preparation is complete"
}
main
