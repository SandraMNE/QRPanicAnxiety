# load .env file if exists
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst) # arg expansion
  # set -o allexport; source .env; set +o allexpor
  # export "$(grep -vE "^(#.*|\s*)$" .env)"
fi

# execute with "source activate.sh" or ". activate.sh"
echo "Activating env [${PRJ_ENV}]"
conda deactivate
conda activate $PRJ_ENV
export PRJ_HOME=$(pwd)
#export GUILD_HOME=./tracking/.guild
#export HF_HOME=~/scratch/cache/huggingface/
#guild check
#echo "*** Check tha [guild_home] is correctly set ***"

#echo "==============================================="
#echo "*** Available operations (execute _guild help_ for more info) ***"
#guild ops 

