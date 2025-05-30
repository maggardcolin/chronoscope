#ALL 100 QUBIT 



#MESH
edges_mesh = [
    # horizontal edges
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
    (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39),
    (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49),
    (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59),
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69),
    (70, 71), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79),
    (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89),
    (90, 91), (91, 92), (92, 93), (93, 94), (94, 95), (95, 96), (96, 97), (97, 98), (98, 99),

    # vertical edges
    (0, 10), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19),
    (10, 20), (11, 21), (12, 22), (13, 23), (14, 24), (15, 25), (16, 26), (17, 27), (18, 28), (19, 29),
    (20, 30), (21, 31), (22, 32), (23, 33), (24, 34), (25, 35), (26, 36), (27, 37), (28, 38), (29, 39),
    (30, 40), (31, 41), (32, 42), (33, 43), (34, 44), (35, 45), (36, 46), (37, 47), (38, 48), (39, 49),
    (40, 50), (41, 51), (42, 52), (43, 53), (44, 54), (45, 55), (46, 56), (47, 57), (48, 58), (49, 59),
    (50, 60), (51, 61), (52, 62), (53, 63), (54, 64), (55, 65), (56, 66), (57, 67), (58, 68), (59, 69),
    (60, 70), (61, 71), (62, 72), (63, 73), (64, 74), (65, 75), (66, 76), (67, 77), (68, 78), (69, 79),
    (70, 80), (71, 81), (72, 82), (73, 83), (74, 84), (75, 85), (76, 86), (77, 87), (78, 88), (79, 89),
    (80, 90), (81, 91), (82, 92), (83, 93), (84, 94), (85, 95), (86, 96), (87, 97), (88, 98), (89, 99)
]


#IBM STYLE


#HEAVY HEX
edges_heavy_hex = [
    # Row 0
    (0, 1), (0, 10),
    (2, 3), (2, 12),
    (4, 5), (4, 14),
    (6, 7), (6, 16),
    (8, 9), (8, 18),
    
    # Row 1
    (11, 12), (11, 21),
    (13, 14), (13, 23),
    (15, 16), (15, 25),
    (17, 18), (17, 27),
    
    # Row 2
    (20, 21), (20, 30),
    (22, 23), (22, 32),
    (24, 25), (24, 34),
    (26, 27), (26, 36),
    (28, 29), (28, 38),
    
    # Row 3
    (31, 32), (31, 41),
    (33, 34), (33, 43),
    (35, 36), (35, 45),
    (37, 38), (37, 47),
    
    # Row 4
    (40, 41), (40, 50),
    (42, 43), (42, 52),
    (44, 45), (44, 54),
    (46, 47), (46, 56),
    (48, 49), (48, 58),
    
    # Row 5
    (51, 52), (51, 61),
    (53, 54), (53, 63),
    (55, 56), (55, 65),
    (57, 58), (57, 67),
    
    # Row 6
    (60, 61), (60, 70),
    (62, 63), (62, 72),
    (64, 65), (64, 74),
    (66, 67), (66, 76),
    (68, 69), (68, 78),
    
    # Row 7
    (71, 72), (71, 81),
    (73, 74), (73, 83),
    (75, 76), (75, 85),
    (77, 78), (77, 87),
    
    # Row 8
    (80, 81), (80, 90),
    (82, 83), (82, 92),
    (84, 85), (84, 94),
    (86, 87), (86, 96),
    (88, 89), (88, 98),
    
    # Row 9
    (91, 92),
    (93, 94),
    (95, 96),
    (97, 98)
]

#TRAPPED ION
# 20 x 5
edges_trapped_ion_5_20 = [
    # Group 0
    (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    # Group 1
    (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9),
    # Group 2
    (10, 11), (10, 12), (10, 13), (10, 14), (11, 12), (11, 13), (11, 14), (12, 13), (12, 14), (13, 14),
    # Group 3
    (15, 16), (15, 17), (15, 18), (15, 19), (16, 17), (16, 18), (16, 19), (17, 18), (17, 19), (18, 19),
    # Group 4
    (20, 21), (20, 22), (20, 23), (20, 24), (21, 22), (21, 23), (21, 24), (22, 23), (22, 24), (23, 24),
    # Group 5
    (25, 26), (25, 27), (25, 28), (25, 29), (26, 27), (26, 28), (26, 29), (27, 28), (27, 29), (28, 29),
    # Group 6
    (30, 31), (30, 32), (30, 33), (30, 34), (31, 32), (31, 33), (31, 34), (32, 33), (32, 34), (33, 34),
    # Group 7
    (35, 36), (35, 37), (35, 38), (35, 39), (36, 37), (36, 38), (36, 39), (37, 38), (37, 39), (38, 39),
    # Group 8
    (40, 41), (40, 42), (40, 43), (40, 44), (41, 42), (41, 43), (41, 44), (42, 43), (42, 44), (43, 44),
    # Group 9
    (45, 46), (45, 47), (45, 48), (45, 49), (46, 47), (46, 48), (46, 49), (47, 48), (47, 49), (48, 49),
    # Group 10
    (50, 51), (50, 52), (50, 53), (50, 54), (51, 52), (51, 53), (51, 54), (52, 53), (52, 54), (53, 54),
    # Group 11
    (55, 56), (55, 57), (55, 58), (55, 59), (56, 57), (56, 58), (56, 59), (57, 58), (57, 59), (58, 59),
    # Group 12
    (60, 61), (60, 62), (60, 63), (60, 64), (61, 62), (61, 63), (61, 64), (62, 63), (62, 64), (63, 64),
    # Group 13
    (65, 66), (65, 67), (65, 68), (65, 69), (66, 67), (66, 68), (66, 69), (67, 68), (67, 69), (68, 69),
    # Group 14
    (70, 71), (70, 72), (70, 73), (70, 74), (71, 72), (71, 73), (71, 74), (72, 73), (72, 74), (73, 74),
    # Group 15
    (75, 76), (75, 77), (75, 78), (75, 79), (76, 77), (76, 78), (76, 79), (77, 78), (77, 79), (78, 79),
    # Group 16
    (80, 81), (80, 82), (80, 83), (80, 84), (81, 82), (81, 83), (81, 84), (82, 83), (82, 84), (83, 84),
    # Group 17
    (85, 86), (85, 87), (85, 88), (85, 89), (86, 87), (86, 88), (86, 89), (87, 88), (87, 89), (88, 89),
    # Group 18
    (90, 91), (90, 92), (90, 93), (90, 94), (91, 92), (91, 93), (91, 94), (92, 93), (92, 94), (93, 94),
    # Group 19
    (95, 96), (95, 97), (95, 98), (95, 99), (96, 97), (96, 98), (96, 99), (97, 98), (97, 99), (98, 99),

    # Inter-group connections
    (4, 5), (9, 10), (14, 15), (19, 20), (24, 25),
    (29, 30), (34, 35), (39, 40), (44, 45), (49, 50),
    (54, 55), (59, 60), (64, 65), (69, 70), (74, 75),
    (79, 80), (84, 85), (89, 90), (94, 95), (99, 0)
]

edges_trapped_ion_10_10 = [
    # Group 0 (0–9)
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
    (5, 6), (5, 7), (5, 8), (5, 9),
    (6, 7), (6, 8), (6, 9),
    (7, 8), (7, 9),
    (8, 9),

    # Group 1 (10–19)
    (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19),
    (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19),
    (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19),
    (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19),
    (14, 15), (14, 16), (14, 17), (14, 18), (14, 19),
    (15, 16), (15, 17), (15, 18), (15, 19),
    (16, 17), (16, 18), (16, 19),
    (17, 18), (17, 19),
    (18, 19),

    # Group 2 (20–29)
    (20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (20, 27), (20, 28), (20, 29),
    (21, 22), (21, 23), (21, 24), (21, 25), (21, 26), (21, 27), (21, 28), (21, 29),
    (22, 23), (22, 24), (22, 25), (22, 26), (22, 27), (22, 28), (22, 29),
    (23, 24), (23, 25), (23, 26), (23, 27), (23, 28), (23, 29),
    (24, 25), (24, 26), (24, 27), (24, 28), (24, 29),
    (25, 26), (25, 27), (25, 28), (25, 29),
    (26, 27), (26, 28), (26, 29),
    (27, 28), (27, 29),
    (28, 29),

    # Group 3 (30–39)
    (30, 31), (30, 32), (30, 33), (30, 34), (30, 35), (30, 36), (30, 37), (30, 38), (30, 39),
    (31, 32), (31, 33), (31, 34), (31, 35), (31, 36), (31, 37), (31, 38), (31, 39),
    (32, 33), (32, 34), (32, 35), (32, 36), (32, 37), (32, 38), (32, 39),
    (33, 34), (33, 35), (33, 36), (33, 37), (33, 38), (33, 39),
    (34, 35), (34, 36), (34, 37), (34, 38), (34, 39),
    (35, 36), (35, 37), (35, 38), (35, 39),
    (36, 37), (36, 38), (36, 39),
    (37, 38), (37, 39),
    (38, 39),

    # Group 4 (40–49)
    (40, 41), (40, 42), (40, 43), (40, 44), (40, 45), (40, 46), (40, 47), (40, 48), (40, 49),
    (41, 42), (41, 43), (41, 44), (41, 45), (41, 46), (41, 47), (41, 48), (41, 49),
    (42, 43), (42, 44), (42, 45), (42, 46), (42, 47), (42, 48), (42, 49),
    (43, 44), (43, 45), (43, 46), (43, 47), (43, 48), (43, 49),
    (44, 45), (44, 46), (44, 47), (44, 48), (44, 49),
    (45, 46), (45, 47), (45, 48), (45, 49),
    (46, 47), (46, 48), (46, 49),
    (47, 48), (47, 49),
    (48, 49),

    # Group 5 (50–59)
    (50, 51), (50, 52), (50, 53), (50, 54), (50, 55), (50, 56), (50, 57), (50, 58), (50, 59),
    (51, 52), (51, 53), (51, 54), (51, 55), (51, 56), (51, 57), (51, 58), (51, 59),
    (52, 53), (52, 54), (52, 55), (52, 56), (52, 57), (52, 58), (52, 59),
    (53, 54), (53, 55), (53, 56), (53, 57), (53, 58), (53, 59),
    (54, 55), (54, 56), (54, 57), (54, 58), (54, 59),
    (55, 56), (55, 57), (55, 58), (55, 59),
    (56, 57), (56, 58), (56, 59),
    (57, 58), (57, 59),
    (58, 59),

    # Group 6 (60–69)
    (60, 61), (60, 62), (60, 63), (60, 64), (60, 65), (60, 66), (60, 67), (60, 68), (60, 69),
    (61, 62), (61, 63), (61, 64), (61, 65), (61, 66), (61, 67), (61, 68), (61, 69),
    (62, 63), (62, 64), (62, 65), (62, 66), (62, 67), (62, 68), (62, 69),
    (63, 64), (63, 65), (63, 66), (63, 67), (63, 68), (63, 69),
    (64, 65), (64, 66), (64, 67), (64, 68), (64, 69),
    (65, 66), (65, 67), (65, 68), (65, 69),
    (66, 67), (66, 68), (66, 69),
    (67, 68), (67, 69),
    (68, 69),

    # Group 7 (70–79)
    (70, 71), (70, 72), (70, 73), (70, 74), (70, 75), (70, 76), (70, 77), (70, 78), (70, 79),
    (71, 72), (71, 73), (71, 74), (71, 75), (71, 76), (71, 77), (71, 78), (71, 79),
    (72, 73), (72, 74), (72, 75), (72, 76), (72, 77), (72, 78), (72, 79),
    (73, 74), (73, 75), (73, 76), (73, 77), (73, 78), (73, 79),
    (74, 75), (74, 76), (74, 77), (74, 78), (74, 79),
    (75, 76), (75, 77), (75, 78), (75, 79),
    (76, 77), (76, 78), (76, 79),
    (77, 78), (77, 79),
    (78, 79),

    # Group 8 (80–89)
    (80, 81), (80, 82), (80, 83), (80, 84), (80, 85), (80, 86), (80, 87), (80, 88), (80, 89),
    (81, 82), (81, 83), (81, 84), (81, 85), (81, 86), (81, 87), (81, 88), (81, 89),
    (82, 83), (82, 84), (82, 85), (82, 86), (82, 87), (82, 88), (82, 89),
    (83, 84), (83, 85), (83, 86), (83, 87), (83, 88), (83, 89),
    (84, 85), (84, 86), (84, 87), (84, 88), (84, 89),
    (85, 86), (85, 87), (85, 88), (85, 89),
    (86, 87), (86, 88), (86, 89),
    (87, 88), (87, 89),
    (88, 89),

    # Group 9 (90–99)
    (90, 91), (90, 92), (90, 93), (90, 94), (90, 95), (90, 96), (90, 97), (90, 98), (90, 99),
    (91, 92), (91, 93), (91, 94), (91, 95), (91, 96), (91, 97), (91, 98), (91, 99),
    (92, 93), (92, 94), (92, 95), (92, 96), (92, 97), (92, 98), (92, 99),
    (93, 94), (93, 95), (93, 96), (93, 97), (93, 98), (93, 99),
    (94, 95), (94, 96), (94, 97), (94, 98), (94, 99),
    (95, 96), (95, 97), (95, 98), (95, 99),
    (96, 97), (96, 98), (96, 99),
    (97, 98), (97, 99),
    (98, 99),

    # Inter-group connections (ring)
    (9, 10), (19, 20), (29, 30), (39, 40), (49, 50),
    (59, 60), (69, 70), (79, 80), (89, 90), (99, 0)
]




def get_map(map_id):
    if map_id==1:
        return edges_trapped_ion_5_20
    if map_id==2:
        return edges_heavy_hex
    if map_id==3:
        return edges_mesh
    if map_id==4:
        return edges_trapped_ion_10_10
