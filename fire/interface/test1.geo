Point(1) = {-6,  2, 0, 0.5};
Point(2) = {-6, -2, 0, 0.5};
Point(3) = { 6, -2, 0, 0.5};
Point(4) = { 6,  2, 0, 0.5};
Point(5) = { 0,  0, 0, 0.1};
Point(6) = { 1,  0, 0, 0.1};
Point(7) = {-1,  0, 0, 0.1};
Point(8) = { 0,  1, 0, 0.1};
Point(9) = { 0, -1, 0, 0.1};
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};
Circle(5) = {8, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 7};
Circle(8) = {7, 5, 8};
Curve Loop( 9) = {1, 2, 3, 4};
Curve Loop(10) = {8, 5, 6, 7};
Plane Surface(1) = {9, 10};
Plane Surface(2) = {10};
Physical Curve("HorEdges", 11) = {1, 3};
Physical Curve("VerEdges", 12) = {2, 4};
Physical Curve("Circle", 13) = {8,5,6,7};
Physical Surface("PunchedDom", 3) = {1};
Physical Surface("Disc", 4) = {2};


