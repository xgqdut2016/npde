lr1 = 5e-1;
lr2 = 1e-1;

Point(1) = {-1,1,0,lr1};
Point(2) = {-1,-1,0,lr1};
Point(3) = {1,-1,0,lr1};
Point(4) = {1,1,0,lr1};

Point(5) = {0.5,1,0,lr2};
Point(6) = {0,1,0,lr2};
Point(7) = {-0.5,1,0,lr2};

Line(9) = {1,2};
Line(10) = {2,3};
Line(11) = {3,4};
Line(12) = {4,5};
Line(13) = {5,6};
Line(14) = {6,7};

Line(15) = {7,1};

Curve Loop(16) = {9,10,11,12,13,14,15};

Plane Surface(1) = {16};

Physical Curve("inflow",17) = {9};
Physical Curve("outflow",18) = {11};
Physical Curve("wall",19) = {10,12,15};
Physical Curve("free",20) = {13,14};

Physical Surface("PunchedDom",21) = {1};

