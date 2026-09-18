"""
knn_C.py

Cau c - Chuong trinh Python cai dat KNN "from scratch" (KHONG dung sklearn):
  - Nhap ten diem (A-F) hoac toa do tuy y, gia tri k, va loai khoang cach
    (euclidean / manhattan).
  - In ra tung buoc tinh toan chi tiet:
      1) Bang khoang cach toi ca 31 diem huan luyen, da sap xep tang dan.
      2) k lang gieng gan nhat duoc chon.
      3) Phan phoi phieu bau theo lop.
      4) Nhan du doan cuoi cung (co neu ro cach xu ly hoa phieu neu co).
  - Demo theo de bai: diem E, k=5, chay ca 2 loai khoang cach.

"""

import math

# Moi phan tu: (x, y, nhan_lop)
TRAINING_DATA = [
    # Tam giac do (8 diem)
    (3, 1, "Tam giac do"), (6, 2, "Tam giac do"), (1, 2, "Tam giac do"), (2, 4, "Tam giac do"),
    (6, 5, "Tam giac do"), (4, 7, "Tam giac do"), (3, 9, "Tam giac do"), (1, 9, "Tam giac do"),
    # Vuong xanh (9 diem)
    (8, 1, "Vuong xanh"), (9, 2, "Vuong xanh"), (5, 3, "Vuong xanh"), (8, 3, "Vuong xanh"),
    (8, 5, "Vuong xanh"), (1, 7, "Vuong xanh"), (3, 7, "Vuong xanh"), (7, 8, "Vuong xanh"),
    (9, 9, "Vuong xanh"),
    # Sao xanh la (5 diem)
    (2, 1, "Sao xanh la"), (8, 2, "Sao xanh la"), (4, 4, "Sao xanh la"), (8, 7, "Sao xanh la"),
    (4, 8, "Sao xanh la"),
    # Tim den (9 diem)
    (5, 1, "Tim den"), (1, 1, "Tim den"), (2, 2, "Tim den"), (4, 2, "Tim den"), (1, 3, "Tim den"),
    (3, 3, "Tim den"), (2, 5, "Tim den"), (3, 5, "Tim den"), (4, 5, "Tim den"),
]
assert len(TRAINING_DATA) == 31, "Phai co dung 31 diem huan luyen"

# 6 diem chua co nhan can phan loai
QUERY_POINTS = {
    "A": (2, 8),
    "B": (6, 7),
    "C": (7, 5),
    "D": (2, 3),
    "E": (7, 2),
    "F": (4, 1),
}

VALID_METRICS = ("euclidean", "manhattan")


# HAM KHOANG CACH

def euclidean_distance(p1, p2):
    """Khoang cach Euclidean giua 2 diem (x1,y1) va (x2,y2)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def manhattan_distance(p1, p2):
    """Khoang cach Manhattan (khoang cach khoi) giua 2 diem."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_distance_function(metric):
    metric = metric.strip().lower()
    if metric not in VALID_METRICS:
        raise ValueError(f"Metric khong hop le: '{metric}'. Chon 'euclidean' hoac 'manhattan'.")
    return euclidean_distance if metric == "euclidean" else manhattan_distance


# THUAT TOAN KNN

def knn_classify(query_point, k, metric="euclidean", training_data=None):
    if training_data is None:
        training_data = TRAINING_DATA
    dist_func = get_distance_function(metric)

    distances = []
    for (x, y, label) in training_data:
        d = dist_func(query_point, (x, y))
        distances.append({"point": (x, y), "label": label, "distance": d})

    distances.sort(key=lambda item: item["distance"])  # tang dan
    neighbors = distances[:k]

    vote_count = {}
    for n in neighbors:
        vote_count[n["label"]] = vote_count.get(n["label"], 0) + 1

    max_votes = max(vote_count.values())
    tied_labels = [lab for lab, c in vote_count.items() if c == max_votes]
    tie_broken = len(tied_labels) > 1

    if not tie_broken:
        predicted_label = tied_labels[0]
    else:
        best_label, best_dist = None, float("inf")
        for n in neighbors:
            if n["label"] in tied_labels and n["distance"] < best_dist:
                best_dist, best_label = n["distance"], n["label"]
        predicted_label = best_label

    return {
        "query_point": query_point,
        "k": k,
        "metric": metric,
        "sorted_distances": distances,
        "neighbors": neighbors,
        "vote_count": vote_count,
        "tied_labels": tied_labels if tie_broken else None,
        "tie_broken": tie_broken,
        "predicted_label": predicted_label,
    }


# IN VERBOSE TUNG BUOC

def _print_line(char="-", length=78):
    print(char * length)


def knn_classify_verbose(point_name_or_coord, k, metric, training_data=None):
    """
    Giong knn_classify() nhung IN RA TUNG BUOC ra console:
      Buoc 1: bang khoang cach da sap xep toi ca 31 diem
      Buoc 2: k lang gieng gan nhat duoc chon
      Buoc 3: phan phoi phieu bau
      Buoc 4: nhan du doan cuoi cung
    Tra ve cung dict ket qua nhu knn_classify().
    """
    if isinstance(point_name_or_coord, str):
        name = point_name_or_coord.upper()
        if name not in QUERY_POINTS:
            raise ValueError(f"Khong tim thay diem '{point_name_or_coord}'. Chon trong A-F.")
        coord = QUERY_POINTS[name]
        display_name = name
    else:
        coord = point_name_or_coord
        display_name = str(coord)

    result = knn_classify(coord, k, metric, training_data)

    _print_line("=")
    print(f"PHAN LOAI DIEM {display_name} = {coord}   |   k = {k}   |   metric = {metric}")
    _print_line("=")

    print(f"\n[Buoc 1] Bang khoang cach toi toan bo {len(result['sorted_distances'])} diem huan luyen "
          f"(da sap xep tang dan):")
    _print_line()
    print(f"{'STT':<5}{'Diem':<12}{'Nhan':<15}{'Khoang cach':<14}{'Trong k-NN?'}")
    _print_line()
    for i, item in enumerate(result["sorted_distances"]):
        marker = "  <-- chon" if i < k else ""
        print(f"{i + 1:<5}{str(item['point']):<12}{item['label']:<15}{item['distance']:<14.4f}{marker}")
    _print_line()

    print(f"\n[Buoc 2] {k} lang gieng gan nhat:")
    for i, n in enumerate(result["neighbors"]):
        print(f"  {i + 1}. Diem {n['point']}  ->  Nhan: {n['label']}  (khoang cach = {n['distance']:.4f})")

    print(f"\n[Buoc 3] Phan phoi phieu bau trong so {k} lang gieng:")
    for label, count in sorted(result["vote_count"].items(), key=lambda x: -x[1]):
        bar = "*" * count
        print(f"  {label:<15}: {count} phieu   {bar}")

    if result["tie_broken"]:
        print(f"\n  >> HOA PHIEU giua cac lop: {result['tied_labels']}")
        print(f"  >> Quy tac pha hoa: chon lop co lang gieng GAN NHAT trong so cac lop dang hoa.")

    print(f"\n[Buoc 4] KET QUA: Diem {display_name} duoc du doan thuoc lop --> {result['predicted_label']} <--")
    _print_line("=")
    print()

    return result


# DEMO: diem E, k=5, ca 2 loai khoang cach

def run_required_demo():
    print("\n" + "#" * 78)
    print("# DEMO BAT BUOC (theo de bai): diem E, k = 5, chay ca 2 loai khoang cach")
    print("#" * 78 + "\n")
    knn_classify_verbose("E", 5, "euclidean")
    knn_classify_verbose("E", 5, "manhattan")


# Input (nhap ten diem + k + loai khoang cach)

def _parse_point_input(raw):
    """Cho phep nhap ten diem (A-F) hoac toa do dang 'x,y'."""
    raw = raw.strip()
    if "," in raw:
        try:
            x_str, y_str = raw.split(",")
            return (float(x_str.strip()), float(y_str.strip()))
        except ValueError:
            raise ValueError("Toa do khong hop le. Nhap dang: x,y (vi du: 5,5)")
    return raw.upper()


def interactive_mode():
    print("\n" + "#" * 78)
    print("# CHE DO NHAP TUONG TAC")
    print("# - Nhap ten diem (A, B, C, D, E, F) hoac toa do dang 'x,y' (vi du: 5,5)")
    print("# - Nhap 'q' de thoat")
    print("#" * 78)

    while True:
        try:
            raw_point = input("\nNhap ten diem hoac toa do (A-F / x,y / q de thoat): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKet thuc chuong trinh.")
            break

        if raw_point.lower() in ("q", "quit", "exit"):
            print("Ket thuc chuong trinh.")
            break

        try:
            point = _parse_point_input(raw_point)
        except ValueError as e:
            print(f"Loi: {e}")
            continue

        try:
            k_raw = input("Nhap k (vi du: 5): ").strip()
            k = int(k_raw)
            if k <= 0:
                raise ValueError
        except (ValueError, EOFError, KeyboardInterrupt):
            print("Loi: k phai la so nguyen duong.")
            continue

        try:
            metric = input("Nhap loai khoang cach (euclidean / manhattan): ").strip().lower()
            get_distance_function(metric)  # validate
        except (ValueError, EOFError, KeyboardInterrupt) as e:
            print(f"Loi: {e}")
            continue

        try:
            knn_classify_verbose(point, k, metric)
        except ValueError as e:
            print(f"Loi: {e}")


if __name__ == "__main__":
    run_required_demo()
    interactive_mode()
