#!/bin/sh

parse_xml()
{
  xml_file="$1"
  out_dir="$2"

  xmllint --xpath "//dataset/images/image/@file" "$xml_file" | xargs | tr ' ' '\n' | cut -f2 -d'=' | while read f
  do
    echo $f
    out_file="${f#*/}"
    out_file="${out_file%.jpg}.pts"
    out_file="${out_dir%/}/${out_file}"
    echo "version: 1" >"$out_file"
    echo "n_points:  5" >>"$out_file"
    echo "{" >>"$out_file"
    xmllint --xpath "//dataset/images/image[@file='$f']/box[1]/part" "$xml_file" | sed 's/>/\n/g' | sed -E 's/.*x=\"([0-9]+)\" y=\"([0-9]+)\".*/\1 \2/g' >>"$out_file"
    echo "}" >>"$out_file"
  done
}

test -e ./points || mkdir ./points

parse_xml ./test_cleaned.xml ./points
parse_xml ./train_cleaned.xml ./points

( cd ./images; ls $PWD/* ) >./images_train.txt
( cd ./points; ls $PWD/* ) >./points_train.txt
