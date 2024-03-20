# load .env file if exists
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst) # arg expansion
fi

# execute with "source activate.sh" or ". activate.sh"
echo "Activating env [${PRJ_ENV}]"
conda deactivate
conda activate $PRJ_ENV
export PRJ_HOME=$(pwd)

