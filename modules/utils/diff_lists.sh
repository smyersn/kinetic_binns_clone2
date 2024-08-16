diff_lists() {
    local list1=("$1")
    local list2=("$2")
    # Sort the lists and compare them
    if [ "$(printf "%s\n" "${list1[@]}" | sort)" != "$(printf "%s\n" "${list2[@]}" | sort)" ]; then
        return 0  # Lists are not equivalent
    else
        return 1  # Lists are equivalent
    fi
}