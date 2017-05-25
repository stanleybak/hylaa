% Plot reachable region (Code generated by Hylaa using PLOT_MATLAB)
h = figure(1);
set(h, 'Position', [200 200 800 600]);
hold on;

% data
reachSet = [
    {'loc1', [0.666666666667 1.0 0.666666666667], [0.0 1.0 0.0], 0, {
    [-6.0, 0.0;-5.0, 0.0;-5.0, 1.0;-6.0, 1.0;-6.0, 0.0;]
    [-5.76395979227, 1.16841240151;-4.8032998269, 0.973677001257;-4.60856442664, 1.93433696664;-5.56922439202, 2.12907236689;-5.76395979227, 1.16841240151;]
    [-5.30967409843, 2.24489409734;-4.42472841536, 1.87074508112;-4.05057939914, 2.75569076419;-4.93552508221, 3.12983978042;-5.30967409843, 2.24489409734;]
    [-4.663630824, 3.19056155458;-3.88635902, 2.65880129548;-3.35459876091, 3.43607309948;-4.13187056491, 3.96783335858;-4.663630824, 3.19056155458;]
    [-3.85884804594, 3.97321877453;-3.21570670495, 3.31101564544;-2.55350357586, 3.95415698643;-3.19664491685, 4.61636011552;-3.85884804594, 3.97321877453;]
    [-2.93331441658, 4.56836649686;-2.44442868048, 3.80697208072;-1.68303426434, 4.29585781681;-2.17192000043, 5.05725223295;-2.93331441658, 4.56836649686;]
    [-1.92829498226, 4.95986682843;-1.60691248522, 4.13322235703;-0.780268013813, 4.45460485407;-1.10165051086, 5.28124932548;-1.92829498226, 4.95986682843;]
    [-0.886573946255, 5.14025278463;-0.738811621879, 4.28354398719;0.117897175559, 4.43130631157;-0.0298651488164, 5.28801510901;-0.886573946255, 5.14025278463;]
    [0.124410970196, 4.25890194001;0.976191358197, 4.23401974597;1.00107355224, 5.08580013397;0.149293164235, 5.11068232801;0.124410970196, 4.25890194001;]
    [0.94887567066, 4.06712932675;1.76230153601, 3.87735419262;1.95207667014, 4.69078005797;1.13865080479, 4.8805551921;0.94887567066, 4.06712932675;]
    [1.70356095253, 3.72234857816;2.44803066816, 3.38163638765;2.78874285867, 4.12610610328;2.04427314304, 4.46681829379;1.70356095253, 3.72234857816;]
    [2.36141585778, 3.24416760554;3.01024937888, 2.77188443398;3.48253255044, 3.42071795509;2.83369902933, 3.89300112664;2.36141585778, 3.24416760554;]
    [2.90027201392, 2.65669067461;3.43161014885, 2.07663627183;4.01166455163, 2.60797440675;3.48032641671, 3.18802880954;2.90027201392, 2.65669067461;]
    [3.30352695369, 1.98739070775;3.70100509524, 1.32668531701;4.36171048598, 1.72416345856;3.96423234443, 2.3848688493;3.30352695369, 1.98739070775;]
    [3.56058142667, 1.2658930105;3.81376002877, 0.553776725164;4.5258763141, 0.806955327263;4.272697712, 1.5190716126;3.56058142667, 1.2658930105;]
    [3.66702220396, 0.522721342859;3.77156647254, -0.210683097933;4.50497091333, -0.106138829362;4.40042664476, 0.627265611431;3.66702220396, 0.522721342859;]
    [3.58216539403, -0.936852361825;4.30707613757, -0.979240685483;4.34946446122, -0.254329941946;3.62455371769, -0.211941618288;3.58216539403, -0.936852361825;]
    [3.25880451084, -1.59757101426;3.94694272502, -1.77945757428;4.12882928503, -1.09131936011;3.44069107086, -0.909432800089;3.25880451084, -1.59757101426;]
    [2.81949934785, -2.16932713434;3.44514641991, -2.47806314679;3.75388243237, -1.85241607473;3.12823536031, -1.54368006228;2.81949934785, -2.16932713434;]
    [2.28613532878, -2.6330420496;2.82704758623, -3.05146800804;3.24547354466, -2.51055575059;2.70456128722, -2.09212979216;2.28613532878, -2.6330420496;]
    [1.68345217403, -2.97464961832;2.12160257743, -3.4819494613;2.62890242042, -3.0437990579;2.19075201701, -2.53649921492;1.68345217403, -2.97464961832;]
    [1.0379554949, -3.18545454471;1.36007980333, -3.75812059196;1.93274585059, -3.43599628353;1.61062154216, -2.86333023627;1.0379554949, -3.18545454471;]
    [0.376801496714, -3.26226533743;0.574735066753, -3.87513169091;1.18760142023, -3.67719812087;0.989667850193, -3.06433176739;0.376801496714, -3.26226533743;]
    [-0.273300474786, -3.20730429226;-0.202500401004, -3.83460513595;0.424800442691, -3.76380506217;0.35400036891, -3.13650421848;-0.273300474786, -3.20730429226;]
    [-0.941267473126, -3.64431761104;-0.324857388279, -3.69846050908;-0.270714490233, -3.08205042424;-0.88712457508, -3.02790752619;-0.941267473126, -3.64431761104;]
    [-1.6139156848, -3.31765189778;-1.03229873693, -3.48970168727;-0.860248947439, -2.90808473939;-1.44186589532, -2.73603494991;-1.6139156848, -3.31765189778;]
    [-2.19648850005, -2.8728487998;-1.67125657305, -3.15139156197;-1.39271381088, -2.62615963498;-1.91794573787, -2.3476168728;-2.19648850005, -2.8728487998;]
    [-2.66952396153, -2.33209671422;-2.21919681972, -2.70196285084;-1.8493306831, -2.25163570904;-2.29965782491, -1.88176957242;-2.66952396153, -2.33209671422;]
    [-3.01864662652, -1.72050108261;-2.658061407, -2.16351131711;-2.2150511725, -1.8029260976;-2.57563639202, -1.3599158631;-3.01864662652, -1.72050108261;]
    [-3.23493546332, -1.06497909393;-2.97480546259, -1.56078000436;-2.47900455216, -1.30065000363;-2.73913455289, -0.804849093201;-3.23493546332, -1.06497909393;]
    [-3.31506214685, -0.393126265495;-3.1617156674, -0.920078876728;-2.63476305617, -0.766732397273;-2.78810953562, -0.23977978604;-3.31506214685, -0.393126265495;]
    [-3.26120304944, 0.267899427686;-3.21650557692, -0.268184835135;-2.6804213141, -0.223487362612;-2.72511878662, 0.312596900208;-3.26120304944, 0.267899427686;]
    [-3.14219315734, 0.368733108981;-2.61849429778, 0.307277590817;-2.55703877962, 0.830976450374;-3.08073763918, 0.892431968538;-3.14219315734, 0.368733108981;]
    [-2.94677373918, 0.966123366892;-2.45564478265, 0.805102805743;-2.2946242215, 1.29623176227;-2.78575317803, 1.45725232342;-2.94677373918, 0.966123366892;]
    [-2.64270906711, 1.50195717779;-2.20225755593, 1.25163098149;-1.95193135963, 1.69208249268;-2.39238287081, 1.94240868897;-2.64270906711, 1.50195717779;]
    [-2.24626057663, 1.95749916073;-1.87188381386, 1.63124930061;-1.54563395373, 2.00562606338;-1.92001071651, 2.3318759235;-2.24626057663, 1.95749916073;]
    [-1.77669823666, 2.31791758119;-1.48058186388, 1.93159798433;-1.09426226702, 2.2277143571;-1.3903786398, 2.61403395397;-1.77669823666, 2.31791758119;]
    [-1.25542223942, 2.57271667766;-1.04618519951, 2.14393056472;-0.617399086569, 2.35316760462;-0.826636126472, 2.78195371757;-1.25542223942, 2.57271667766;]
    [-0.705034855641, 2.71598107617;-0.587529046367, 2.26331756348;-0.134865533672, 2.38082337275;-0.252371342945, 2.83348688545;-0.705034855641, 2.71598107617;]
    [-0.148401061165, 2.74642953384;-0.123667550971, 2.2886912782;0.33407070467, 2.3134247884;0.309337194475, 2.77116304404;-0.148401061165, 2.74642953384;]
    [0.326886797735, 2.22273652182;0.771434102099, 2.15735916227;0.836811461646, 2.60190646664;0.392264157282, 2.66728382619;0.326886797735, 2.22273652182;]
    [0.746872583834, 2.0716375424;1.16120009232, 1.92226302564;1.31057460908, 2.33659053412;0.896247100601, 2.48596505088;0.746872583834, 2.0716375424;]
    [1.12091178585, 1.84469669727;1.48985112531, 1.6205143401;1.71403348248, 1.98945367956;1.34509414303, 2.21363603673;1.12091178585, 1.84469669727;]
    [1.4360428508, 1.55384503442;1.74681185769, 1.26663646426;2.03402042785, 1.57740547115;1.72325142096, 1.86461404131;1.4360428508, 1.55384503442;]
    [1.68213754731, 1.21306831362;1.92475121003, 0.876640804162;2.26117871949, 1.11925446689;2.01856505677, 1.45568197635;1.68213754731, 1.21306831362;]
    [1.85218957973, 0.837774404183;2.01974446056, 0.467336488238;2.39018237651, 0.634891369074;2.22262749567, 1.00532928502;1.85218957973, 0.837774404183;]
    [1.9424687456, 0.444129415276;2.03129462866, 0.0556356661547;2.41978837778, 0.14446154921;2.33096249473, 0.532955298331;1.9424687456, 0.444129415276;]
    [1.95253970146, 0.0483898772567;1.96221767692, -0.342118063036;2.35272561721, -0.332440087585;2.34304764176, 0.0580678527081;1.95253970146, 0.0483898772567;]
    [1.8184014763, -0.710772420312;2.19543146868, -0.777520905898;2.26217995427, -0.400490913521;1.88514996189, -0.333742427934;1.8184014763, -0.710772420312;]
    [1.60845294572, -1.03691779809;1.95765227367, -1.17446149211;2.0951959677, -0.825262164165;1.74599663975, -0.687718470137;1.60845294572, -1.03691779809;]
    [1.34325173808, -1.3093481926;1.65192892588, -1.50948239356;1.85206312684, -1.20080520576;1.54338593904, -1.0006710048;1.34325173808, -1.3093481926;]
    [1.03543169843, -1.51941707942;1.29299229733, -1.77178837552;1.54536359343, -1.51422777661;1.28780299453, -1.26185648051;1.03543169843, -1.51941707942;]
    [0.698813460454, -1.66127838403;0.897095986851, -1.95387755556;1.18969515838, -1.75559502916;0.991412631981, -1.46299585764;0.698813460454, -1.66127838403;]
    [0.347812372284, -1.73200736083;0.481315034678, -2.05170830052;0.801015974366, -1.91820563813;0.667513311972, -1.59850469844;0.347812372284, -1.73200736083;]
    [-0.00315375678504, -1.73160151632;0.0628398098457, -2.06472310626;0.395961399784, -1.99872953963;0.329967833154, -1.66560794969;-0.00315375678504, -1.73160151632;]
    [-0.341707025477, -1.99573396103;-0.00883916624318, -1.9972071554;-0.00736597186931, -1.66433929617;-0.340233831103, -1.66286610179;-0.341707025477, -1.99573396103;]
    [-0.716904345486, -1.85067925283;-0.397418403257, -1.91691565337;-0.331182002714, -1.59742971114;-0.650667944943, -1.5311933106;-0.716904345486, -1.85067925283;]
    [-1.04909410144, -1.63826679759;-0.755075320712, -1.76411268438;-0.629229433927, -1.47009390365;-0.923248214656, -1.34424801686;-1.04909410144, -1.63826679759;]
    [-1.32685127065, -1.36952154446;-1.06890585072, -1.54767251958;-0.8907548756, -1.28972709965;-1.14870029553, -1.11157612453;-1.32685127065, -1.36952154446;]
    [-1.54134724386, -1.05725958048;-1.3282417107, -1.27863319893;-1.10686809225, -1.06552766578;-1.31997362541, -0.844154047329;-1.54134724386, -1.05725958048;]
    [-1.6865964751, -0.715512050628;-1.5249838054, -0.969676018194;-1.27081983783, -0.808063348495;-1.43243250753, -0.553899380929;-1.6865964751, -0.715512050628;]
    [-1.75958124802, -0.358923711083;-1.65382115303, -0.634560569921;-1.37818429419, -0.528800474934;-1.48394438918, -0.253163616096;-1.75958124802, -0.358923711083;]
    [-1.76025441625, -0.00215085001641;-1.71233118607, -0.287539381029;-1.42694265506, -0.239616150857;-1.47486588523, 0.045772380155;-1.76025441625, -0.00215085001641;]
    [-1.70096209531, 0.0572240262385;-1.41746841276, 0.0476866885321;-1.40793107505, 0.331180371084;-1.69142475761, 0.340717708791;-1.70096209531, 0.0572240262385;]
    [-1.62290257162, 0.386210453611;-1.35241880968, 0.321842044676;-1.28805040075, 0.592325806612;-1.55853416268, 0.656694215547;-1.62290257162, 0.386210453611;]
    [-1.48384869825, 0.687053423692;-1.23654058187, 0.572544519743;-1.12203167792, 0.819852636118;-1.3693397943, 0.934361540066;-1.48384869825, 0.687053423692;]
    [-1.29168037932, 0.948982567454;-1.0764003161, 0.790818806212;-0.918236554857, 1.00609886943;-1.13351661808, 1.16426263067;-1.29168037932, 0.948982567454;]
    [-1.05606509308, 1.16318539277;-0.880054244231, 0.969321160642;-0.686190012102, 1.14533200949;-0.862200860949, 1.33919624162;-1.05606509308, 1.16318539277;]
    [-0.788006120735, 1.32307897552;-0.656671767279, 1.10256581293;-0.436158604693, 1.23390016639;-0.567492958149, 1.45441332897;-0.788006120735, 1.32307897552;]
    [-0.499355621184, 1.42448170817;-0.41612968432, 1.18706809014;-0.178716066292, 1.27029402701;-0.261942003156, 1.50770764503;-0.499355621184, 1.42448170817;]
    [-0.20231393924, 1.46568477168;-0.168594949367, 1.2214039764;0.0756858459125, 1.25512296627;0.0419668560392, 1.49940376155;-0.20231393924, 1.46568477168;]
    [0.0758881776256, 1.20618531063;0.317125239752, 1.19100767511;0.332302875278, 1.43224473724;0.0910658131507, 1.44742237276;0.0758881776256, 1.20618531063;]
    [0.307789715905, 1.14395582837;0.536580881579, 1.08239788519;0.598138824759, 1.31118905086;0.369347659086, 1.37274699404;0.307789715905, 1.14395582837;]
    [0.518449953754, 1.0390150215;0.726252958054, 0.935325030751;0.829942948805, 1.14312803505;0.622139944504, 1.2468180258;0.518449953754, 1.0390150215;]
    [0.700387137865, 0.897179574748;0.879823052815, 0.757102147175;1.01990048039, 0.936538062124;0.840464565438, 1.0766154897;0.700387137865, 0.897179574748;]
    [0.84754653793, 0.725494301946;0.99264539832, 0.555984994359;1.16215470591, 0.701083854749;1.01705584552, 0.870593162335;0.84754653793, 0.725494301946;]
    [0.955483472684, 0.531905992135;1.06186467111, 0.340809297598;1.25296136565, 0.447190496025;1.14658016722, 0.638287190562;0.955483472684, 0.531905992135;]
    [1.02147565637, 0.324914317665;1.0864585199, 0.120619186391;1.29075365118, 0.185602049924;1.22577078765, 0.389897181198;1.02147565637, 0.324914317665;]
    [1.04456309608, 0.113214686228;1.06720603333, -0.0956979329887;1.27611865255, -0.073054995743;1.2534757153, 0.135857623474;1.04456309608, 0.113214686228;]
    [1.00658633544, -0.299755987796;1.21168970718, -0.318686511009;1.23062023039, -0.113583139277;1.02551685866, -0.0946526160643;1.00658633544, -0.299755987796;]
    [0.908614090038, -0.483981592486;1.10196224553, -0.542108279885;1.16008893293, -0.348760124398;0.966740777438, -0.290633436998;0.908614090038, -0.483981592486;]
    [0.778620824802, -0.641881090589;0.953043333056, -0.735372807057;1.04653504952, -0.560950298803;0.872112541269, -0.467458582336;0.778620824802, -0.641881090589;]
    [0.622992873695, -0.768254525276;0.772347446532, -0.892034515764;0.89612743702, -0.742679942928;0.746772864183, -0.61889995244;0.622992873695, -0.768254525276;]
    [0.448877946464, -0.859350151136;0.568252557366, -1.00734525918;0.716247665413, -0.887970648281;0.596873054511, -0.739975540234;0.448877946464, -0.859350151136;]
    [0.263873160166, -0.912955728844;0.349731680538, -1.07837517054;0.515151122232, -0.992516650167;0.42929260186, -0.827097208472;0.263873160166, -0.912955728844;]
    [0.0757075628514, -0.928425475897;0.125975381703, -1.10405700731;0.301606913113, -1.05378918845;0.25133909426, -0.878157657045;0.0757075628514, -0.928425475897;]
    [-0.108068101766, -0.90664413425;-0.0939795007884, -1.0851552409;0.0845316058661, -1.07106663993;0.0704430048884, -0.892555533273;-0.108068101766, -0.90664413425;]
    [-0.301600509494, -1.02416406585;-0.127368485045, -1.04539214669;-0.106140404205, -0.871160122242;-0.280372428653, -0.849932041401;-0.301600509494, -1.02416406585;]
    [-0.48917656011, -0.925141120089;-0.325932687661, -0.979463234699;-0.271610573051, -0.816219362249;-0.4348544455, -0.761897247639;-0.48917656011, -0.925141120089;]
    [-0.650090089085, -0.793486038872;-0.503846674968, -0.8774604847;-0.41987222914, -0.731217070584;-0.566115643257, -0.647242624756;-0.650090089085, -0.793486038872;]
    [-0.779035367981, -0.635674707893;-0.654897972972, -0.744824370055;-0.54574831081, -0.620686975046;-0.669885705819, -0.511537312884;-0.779035367981, -0.635674707893;]
    [-0.872176480038, -0.458961465367;-0.774177959166, -0.587991125228;-0.645148299305, -0.489992604357;-0.743146820177, -0.360962944496;-0.872176480038, -0.458961465367;]
    [-0.927241089779, -0.271062252695;-0.858224478807, -0.41409966583;-0.715187065673, -0.345083054858;-0.784203676644, -0.202045641724;-0.927241089779, -0.271062252695;]
    [-0.943548822483, -0.0798319700474;-0.905101778084, -0.230682266395;-0.754251481736, -0.192235221996;-0.792698526136, -0.0413849256483;-0.943548822483, -0.0798319700474;]
    [-0.921975697647, 0.107051000772;-0.914417057295, -0.0453518421108;-0.762014214413, -0.037793201759;-0.769572854765, 0.114609641124;-0.921975697647, 0.107051000772;]
    [-0.887275475074, 0.134501693306;-0.739396229228, 0.112084744422;-0.716979280344, 0.259963990267;-0.86485852619, 0.282380939152;-0.887275475074, 0.134501693306;]
    [-0.826177789465, 0.301994358466;-0.688481491221, 0.251661965388;-0.638149098143, 0.389358263632;-0.775845396387, 0.43969065671;-0.826177789465, 0.301994358466;]
    [-0.734866933726, 0.450999974384;-0.612389111438, 0.375833311987;-0.537222449041, 0.498311134274;-0.659700271329, 0.573477796672;-0.734866933726, 0.450999974384;]
    [-0.618131577911, 0.5763622475;-0.515109648259, 0.480301872917;-0.419049273676, 0.583323802569;-0.522071203328, 0.679384177152;-0.618131577911, 0.5763622475;]
    [-0.481576118707, 0.674060256488;-0.401313432256, 0.561716880407;-0.288970056175, 0.641979566858;-0.369232742626, 0.754322942939;-0.481576118707, 0.674060256488;]
    [-0.331367491683, 0.741322637445;-0.276139576402, 0.617768864538;-0.152585803495, 0.672996779818;-0.207813718775, 0.796550552726;-0.331367491683, 0.741322637445;]
    }}
];

% plot all
for i = 1:size(reachSet,1)
    face_color = reachSet{i,2};
    edge_color = reachSet{i,3};
    poly_list = reachSet{i,5};

    for p_index = 1:size(poly_list,1)
        pts = poly_list{p_index};
        h = fill(pts(:,1), pts(:,2), face_color, 'EdgeColor', edge_color);

        reachSet{i,4} = h;  % add handle to reachSet data structure for use in legend
    end
end

% optional legend
if (size(reachSet,1) > 1 && size(reachSet,1) < 10)
    legend([reachSet{:,3}], reachSet{:,1})
end

% labels and such
xlabel('X', 'FontSize', 24, 'FontName', 'Serif', 'Interpreter','LaTex');
ylabel('Y', 'FontSize', 24, 'FontName', 'Serif', 'Interpreter','LaTex');
title('HybridAutomaton', 'FontSize', 32, 'FontName', 'Serif', 'Interpreter','LaTex');
hold off;