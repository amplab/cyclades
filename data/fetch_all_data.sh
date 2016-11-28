for dir in ./*/
do
    echo "Fetching data for directory: " $dir
    cd $dir
    if [ -f get_data.sh ]; then
	sh get_data.sh 2>&1 > /dev/null
    fi
    cd ..
done
