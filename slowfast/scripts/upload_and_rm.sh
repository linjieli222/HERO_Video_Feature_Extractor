startIdx=$1
endIdx=$2
red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
blue=`tput setaf 4`
reset=`tput sgr0`

pathPrefix="/datadrive/linjie/howto100m"
blobFolder="/home/linjie/convaiblobvideolinjli/vision_features/slowfast/howto100m/"
blobVideoFolder="/home/linjie/convaiblobvideolinjli/howto100m/raw_videos/videos"
blob="https://convaistorageglobal.blob.core.windows.net/convaiblobvideolinjli/vision_features/slowfast/howto100m/"
key="?sv=2018-03-28&ss=bfqt&srt=sco&sp=rwdlacup&se=2021-04-02T14:42:44Z&st=2019-08-31T06:42:44Z&spr=https&sig=x8WQrE%2F8oNy%2FvJj%2B9xM9auhmtrb5q8wMNgE56qNunjE%3D"

flag=1
for ((i = $startIdx ; i < $endIdx ; i++)); do
	videoFolder="${blobVideoFolder}/${i}" # "${pathPrefix}_videos/$i/"
	featureFolder="${pathPrefix}_features/$i/"
	if [ -d $videoFolder ] && [ -d $featureFolder ]; then
		# echo "$videoFolder exists, start checking"
		videoFileNum=$(ls -l $videoFolder | wc -l)
		featureFileNum=$(ls -l $featureFolder | wc -l)
		if [ $videoFileNum -eq $featureFileNum ]; then
			if [ ! -d "$blobFolder/$i" ]; then
				# echo "${green}Uploading $featureFolder...${reset}"
				~/azcopy cp $featureFolder $blob$key --recursive --overwrite=false > /dev/null
			else
				blobFeatureFileNum=$(ls -l "$blobFolder/$i" | wc -l)
				if [ ! $featureFileNum -eq $blobFeatureFileNum ]; then
				 	# echo "${yellow}Skip uploading...${reset}"
				 	# rm -rf $featureFolder
				# else
					# echo "${green}Uploading $featureFolder...${reset}"
					 ~/azcopy cp $featureFolder $blob$key --recursive --overwrite=true > /dev/null
				fi
			fi
			# echo "${yellow}$i: $videoFileNum == $featureFileNum${reset}"
			# echo "${yellow}$videoFileNum == $featureFileNum, removing $videoFolder ...${reset}"
			videoLocalCopy="${pathPrefix}_videos/$i/"
			if [ -d $videoLocalCopy ]; then
				# echo "${yellow}removing local copy...${reset}"
				rm -rf $videoLocalCopy
			fi
		else
			flag=0
			echo "${red}$videoFileNum != $featureFileNum. Mismatch for $i...${reset}"
		fi
	else
		if [ -d $videoFolder ] && [ ! -d $featureFolder ]; then
			echo "${red}featureFolder for ${i} does not exists...${reset}"
			flag=0
	        # else
			# echo "${blue}videoFolder for ${i} does not exists...${reset}"
		fi
	fi
done

infoFolder="${pathPrefix}_info"
if [ $flag -eq 1 ]; then
    echo "${green}All videos are processed from ${startIdx} to $(($endIdx -1))${reset}"
    infoFile="${infoFolder}/${startIdx}_$(($endIdx -1)).csv"
	if [ -f $infoFile ]; then
		echo "${green}moving ${infoFile}....${reset}"
		mv  $infoFile ${infoFolder}/done/
	fi
	infoFailedFile="${infoFolder}/${startIdx}_$(($endIdx -1))_failed.txt"
	if [ -f $infoFailedFile ]; then
		echo "${green}moving ${infoFailedFile}....${reset}"
		mv  $infoFailedFile ${infoFolder}/done/
	fi
fi
