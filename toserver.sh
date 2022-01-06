# ./toserver.sh room@15123 local_folder exp_name
# ./toserver.sh room@15123 phasen_torch se_phasen_009

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Need a destination."
  exit -1
fi
site=${1#*@}
user=${1%@*}
rm _data _log -rf
rm *__pycache__* -rf
rm */__pycache__* -rf
# mv exp ../
# # scp -r -P 15044 ./* student@speaker.is99kdf.xyz:~/lhf/work/irm_test/extract_tfrecord
# scp -r -P 15043 ./* room@speaker.is99kdf.xyz:~/work/speech_en_test/c001_se

# mv ../exp ./

if [ "$site" == "p40" ]; then
  echo "To $user@$site:/home/zhangwenbo5/lihongfeng/TorchPHASEN/$2"
  rsync -avh -e "ssh -p 22 -o ProxyCommand='ssh -p 8695 zhangwenbo5@120.92.114.84 -W %h:%p'" --exclude-from='.gitignore' ./$2/* zhangwenbo5@ksai-P40-2:/home/zhangwenbo5/lihongfeng/TorchPHASEN/$3
elif [ "$site" == "v100-3" ]; then
  echo "To $user@$site:/home/zhangwenbo5/lihongfeng/TorchPHASEN/$2"
  rsync -avh -e "ssh -p 22 -o ProxyCommand='ssh -p 8695 zhangwenbo5@120.92.114.84 -W %h:%p'" --exclude-from='.gitignore' ./$2/* zhangwenbo5@ksai-v100-3:/home/zhangwenbo5/lihongfeng/TorchPHASEN/$3
elif [ "$site" == "15123" ] || [ "$site" == "15041" ] || [ "$site" == "15043" ]; then
  echo "To $user@$site:~/worklhf/TorchPHASEN/$2"
  rsync -avh -e 'ssh -p '$site --exclude-from='.gitignore' ./$2/* $user@speaker.is99kdf.xyz:~/worklhf/TorchPHASEN/$3
elif [ "$site" == "15043ali" ]; then
  echo "To $user@$site:~/worklhf/TorchPHASEN/$2"
  rsync -avh -e 'ssh -p 6662' --exclude-from='.gitignore' ./$2/* $user@47.92.169.196:~/worklhf/TorchPHASEN/$3
elif [ "$site" == "bss13001" ]; then
  echo "To $user@$site:~/worklhf/SE_VoiceBankDEMAND/$3"
  rsync -avh -e 'ssh -p 13001' --exclude-from='.gitignore' ./$2/*  $user@10.221.224.210:/root/worklhf/SE_VoiceBankDEMAND/$3
elif [ "$site" == "bss13002" ]; then
  echo "To $user@$site:~/worklhf/SE_VoiceBankDEMAND/$3"
  rsync -avh -e 'ssh -p 13002' --exclude-from='.gitignore' ./$2/* $user@10.221.224.210:/root/worklhf/SE_VoiceBankDEMAND/$3
elif [ "$site" == "bss13003" ]; then
  echo "To $user@$site:~/worklhf/SE_VoiceBankDEMAND/$3"
  rsync -avh -e 'ssh -p 13003' --exclude-from='.gitignore' ./$2/* $user@10.221.224.210:/root/worklhf/SE_VoiceBankDEMAND/$3
elif [ "$site" == "bss13004" ]; then
  echo "To $user@$site:~/worklhf/SE_VoiceBankDEMAND/$3"
  rsync -avh -e 'ssh -p 13004' --exclude-from='.gitignore' ./$2/* $user@10.221.224.210:/root/worklhf/SE_VoiceBankDEMAND/$3
elif [ "$site" == "bss13006" ]; then
  echo "To $user@$site:~/worklhf/SE_VoiceBankDEMAND/$3"
  rsync -avh -e 'ssh -p 13006' --exclude-from='.gitignore' ./$2/* $user@10.221.224.210:/root/worklhf/SE_VoiceBankDEMAND/$3
elif [ "$site" == "bss13007" ]; then
  echo "To $user@$site:~/worklhf/SE_VoiceBankDEMAND/$3"
  rsync -avh -e 'ssh -p 13007' --exclude-from='.gitignore' ./$2/* $user@10.221.224.210:/root/worklhf/SE_VoiceBankDEMAND/$3
elif [ "$site" == "bss13008" ]; then
  echo "To $user@$site:~/worklhf/SE_VoiceBankDEMAND/$3"
  rsync -avh -e 'ssh -p 13008' --exclude-from='.gitignore' ./$2/*  $user@10.221.224.210:/root/worklhf/SE_VoiceBankDEMAND/$3
fi
# -a ：递归到目录，即复制所有文件和子目录。另外，打开归档模式和所有其他选项（相当于 -rlptgoD）
# -v ：详细输出
# -e ssh ：使用 ssh 作为远程 shell，这样所有的东西都被加密
# --exclude='*.out' ：排除匹配模式的文件，例如 *.out 或 *.c 等。

# scp -r -P 15043 room@speaker.is99kdf.xyz:/home/room/work/paper_se_test/pc001_se/exp/rnn_speech_enhancement/nnet_C001/nnet_iter15* ./
# scp -P 15223 room@speaker.is99kdf.xyz:/fast/worklhf/paper_se_test/C_UNIGRU_RealPSM_RelativeLossAFD100/exp/rnn_speech_enhancement/nnet_C_UNIGRU_RealPSM_RelativeLossAFD100/nnet_iter25* ./
