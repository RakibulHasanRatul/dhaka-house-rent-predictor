def construct_features_list(
    beds: float, bath: float, area: float, type_num: float, year: float
) -> list[float]:
    return [beds, bath, area, year, type_num]
