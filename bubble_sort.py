"""Bubble sort: simple, stable, O(n^2) comparison sort.

Runs in-place on a list of comparable items. Early exit when a pass
makes no swaps.
"""


def bubble_sort(arr: list) -> list:
    """Sort *arr* in place (ascending) and return it.

    Args:
        arr: list of comparable items (e.g. ints, strings).

    Returns:
        The same list, sorted in place.
    """
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


if __name__ == "__main__":
    sample = [64, 34, 25, 12, 22, 11, 90]
    print("before:", sample)
    print("after: ", bubble_sort(sample))
