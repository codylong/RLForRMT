cd ../output
for d in */ ; do
    echo ""
    cat  "$d""command.txt"
    echo ""
    cat "$d""scores.txt"
done