solver="../build/cadical"
sat_output="SATISFIABLE"
time_limit_output="UNKNOWN"
k=$1
partition=$2
cnf_dir="../../../datasets/${k}SAT/sc/${partition}-final"
out_file="${k}SAT-${partition}-labels.csv"


if [ -f "$out_file" ]; then
    rm "$out_file"
    echo "Removed existing label file ${out_file}"
fi

echo "Starting solving procedure for all CNF problems in ${cnf_dir}..."
echo ""

echo "cnf_file,sat,assignment" > $out_file
for filename in "$cnf_dir"/*; do
    output=$($solver -t 60 $filename)
    satisfiable=$(echo "$output" | grep -oP '^s \K\S+')
    unknown=$(echo "$output" | grep -oP 'UNKNOWN')

    if [ "$satisfiable" = "$sat_output" ]; then
        satisfiable_bit=1
    elif [ "$unknown" = "$time_limit_output" ]; then
        satisfiable_bit=-1
    else
        satisfiable_bit=0
    fi
    
    savename="${partition}-final/$(basename "$filename")"
    assignments=$(echo "$output" | grep -oP '^v \K[\d -]+')
    assignments=$(echo "$assignments" | tr '\n' ' ')
    echo "$savename,$satisfiable_bit,$assignments" >> $out_file
done

mv $out_file "../../../datasets/${k}SAT/sc/"
echo "Done."
