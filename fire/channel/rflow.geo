lr1 = 5e-1;
lr2 = 1e-1;

L1 = 1.2743;
L2 = 1;
a = 0.06;
b = 0.25701;
h = 0.11598;

Point(1) = {0,0,0,lr1};
Point(2) = {L1 + L2,0,0,lr1};
Point(3) = {L1 + L2,2*h + 2*b + 2*a,0,lr1};
Point(4) = {0,2*h + 2*b + 2*a,0,lr1};
Point(5) = {0,h + 2*b + 2*a,0,lr2};
Point(6) = {L1,h + 2*b + 2*a,0,lr2};
Point(7) = {L1,h + b + 2*a,0,lr2};
Point(8) = {0,h + b + 2*a,0,lr2};
Point(9) = {0,h + b ,0,lr2};
Point(10) = {L1,h + b ,0,lr2};
Point(11) = {L1,h ,0,lr2};
Point(12) = {0,h ,0,lr2};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,1};

Curve Loop(13) = {1,2,3,4,5,6,7,8,9,10,11,12};

Plane Surface(1) = {13};

Physical Curve("inflow",14) = {8};
Physical Curve("outflowu",15) = {4};
Physical Curve("outflowl",16) = {12};
Physical Curve("wall",17) = {1,2,3,6,10};
Physical Curve("free",18) = {5,7,9,11};

Physical Surface("PunchedDom",18) = {1};



