cd ../output
for d in *$0*/ ; do
    echo ""
    cat  "$d""command.txt"
    echo ""
    cat "$d""scores.txt"
done