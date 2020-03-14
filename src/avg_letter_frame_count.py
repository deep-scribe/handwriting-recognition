import data_loader_upper

if __name__ == "__main__":
    xs, ys = data_loader_upper.verified_subjects_calibrated_yprs(
        resampled=False, flatten=False, keep_idx_and_td=False
    )

    total_frame_count = 0
    for x in xs:
        total_frame_count += len(x)
    print(f'Avg frame per letter {total_frame_count / len(ys)}')
