from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np
import torch
import pandas as pd


def coordinate():
    pt = [np.array([np.cos(i * 2 * np.pi / 20), np.sin(i * 2 * np.pi / 20), 1]) for i in range(20)]
    P = np.array(pt)
    V = np.empty((20, 20, 3), dtype=object)  
    for i in range(20):
        for j in range(20):
            V[i, j] = pt[i] + (1 / 4) * (pt[j] - pt[i])
    return P, V

def GRS(seq, P, V, M):
    l_seq = len(seq)
    k = len(M)
    g = []

    for j in range(k):
        c = [np.array([0, 0, 0])]
        d = np.zeros(3)
        y = None

        for i in range(l_seq):
            # 将 x 转换为二维数组，以便与 P 进行操作
            x = np.array([ch == seq[i] for ch in M[j]])[:, np.newaxis]
            if i == 0:
                c.append(c[-1] + np.dot(x.T, P))
            else:
                if np.all(x == 0):
                    d = d * (i - 1) / i
                    c.append(c[-1] + np.array([0, 0, 1]) + d)
                elif y is None or np.all(y == 0):
                    d = d * (i - 1) / i
                    c.append(c[-1] + np.dot(x.T, P) + d)
                else:
                    idx_y, _ = np.where(y == 1)
                    idx_x, _ = np.where(x == 1)
                    d = d * (i - 1) / i + V[idx_y[0], idx_x[0]] / i
                    c.append(c[-1] + np.dot(x.T, P) + d)
            y = x
        g.append(np.vstack(c))
    return g


def ME(W):
    W = W[1:, :]  
    x, y = W.shape
    W_numeric = np.array(W, dtype=float)
    D = pdist(W_numeric)  # 计算距离
    E = squareform(D)  

    sdist = np.zeros((x, x))
    for i in range(x):
        for j in range(i, x):
            if j - i == 1:
                sdist[i, j] = E[i, j]
            elif j - i > 1:
                sdist[i, j] = sdist[i, j - 1] + E[j - 1, j]
            else:
                sdist[i, j] = 0

    sd = sdist + sdist.T
    sdd = sd + np.diag(np.ones(x))
    L = E / sdd
    ME, _ = eigh(L, eigvals=(x - 1, x - 1))  # 计算最大特征值
    return ME[0] / x


def SAD(seq, a):
    len_seq = len(seq)
    len_a = len(a)
    AAC = np.zeros(len_a)
    DPC = np.zeros((len_a, len_a))

    # 计算氨基酸组成 (AAC)
    for i in range(len_a):
        AAC[i] = seq.count(a[i]) / len_seq

    # 计算二肽组成 (DPC)
    if len_seq != 1:
        for i in range(len_a):
            for j in range(len_a):
                c_i = [char == a[i] for char in seq]
                c_j = [char == a[j] for char in seq]
                c_j_shifted = c_j[1:] + [False] 
                DPC[i, j] = sum(np.array(c_j_shifted) * 2 - np.array(c_i) == 1) / (len_seq - 1)

    return AAC, DPC


def FEGS(seq):
    EL = np.zeros(158)
    # 计算 EL 特征
    P, V = coordinate()
    g_p = GRS(seq, P, V, M)
    for u in range(158):
        EL[u] = ME(g_p[u])

    # 计算 AAC 和 DIC 特征
    char = 'ARNDCQEGHILKMFPSTWYV'
    AAC, DIC = SAD(seq, char)
    FD = DIC.flatten()
    FV = np.concatenate((EL, AAC, FD), axis=0)
    return torch.tensor(FV, dtype=torch.float32)  # dimension =343


M = np.array(['DKERNHYGQPWTSVAICFML', 'DKENHRQGYPSTWVCAIFLM',
              'MWFHCALVYTINKQESPDRG', 'KEQDNRSTPGYAHCILVMFW',
              'CENSKRDHGQTVPMYIAWFL', 'PWYIFTARCESGKDQHMLVN',
              'GVIETAKQLRPSMYHFCWND', 'HNWEKDAGFQPSRMTLIYVC',
              'MEAQLPFKDCYGRHWTSINV', 'GASCTDPNVEQHLIMKRFYW',
              'WLIFHVCEYAMNGQRTDKSP', 'VHMIAEFPLTDRNKQSYCGW',
              'ILVWMFAHEQCTRYKSNDGP', 'DGQSETAKRNPICHVLMWFY',
              'KEDPQNRWSHGYTILFAVCM', 'EDKRQPNYSHTWGAVLIFMC',
              'RKDQNEHSTPYCGAMWLVFI', 'RKDENQSGHTAPYVMCLFIW',
              'MLVIWFKHAYCQERTSDGNP', 'YLVFWIMQHTRKCASGEDNP',
              'ILMVWFQKCAHYRETDSNGP', 'FMCVLIWHYTAGNSPDEQKR',
              'GRASTPDNVEQILFCYWKHM', 'GASTPDNVEQILFYCMWKHR',
              'GASPDMFTVNCEYLIQWHKR', 'PGNSYTCVDHWIRFKQLAEM',
              'CIVGFLAMWSTHPYNDEQRK', 'WMHCYFQPRINDETVKLSAG',
              'PGSYTCNVWRDFIHQLKAME', 'PSGNCYTDVHQRWFIKEALM',
              'MCIFWVYAHLNQREGSTKPD', 'CMIFLVWYHATSEDNQRPGK',
              'CIMFWVYLATSHDGQENRPK', 'GASCPTVINDLMQEHKFYWR',
              'GASCTPVDNLIQEHMFKYWR', 'PDCENKQAGSMRLHWTYFIV',
              'MVILFHWACRKQETYNSDGP', 'DSKPNTRQEHGAYWFMCLIV',
              'EDNKAQRGLSHWMFCTPIYV', 'GASPTDNVEQLIKMHRFYCW',
              'ERSKDQCNGAHTVMPYLIFW', 'NSQKTDHRGEAPMVCYILFW',
              'KDNEQSPRTGAHYCVWLIMF', 'WCHMYFQNPDITRKEVSGAL',
              'DREKVQAGCNHPYSTIWFLM', 'CWHQERDPMKAYTVFGSNIL',
              'CHWMQYKNDRPETSFIVGAL', 'KHEDRNCQPWYMSTGFAIVL',
              'RDCHKQENWYPGVASMFTIL', 'WCMYHFINTQVDKPRGELSA',
              'WCYIHFTMVQNRKDPSGLEA', 'MWHCYFRIQNAKPDGVETSL',
              'MWCHYFRIQATDNEVPKGSL', 'ERKHDQNMWPCYSTFGAVIL',
              'WCHMYFNIDTPVQGRASELK', 'HWMCQFYRPKIANTDEVSLG',
              'DKNPESQTRGHAFYMWLIVC', 'KDEPSQNGTRAHYMVCLWFI',
              'CVLMIGAWFTHSPYNEQDRK', 'WIFVCMYLRHGTAPSQNEDK',
              'WFYHRLMIVEQNTKCDPSAG', 'WFYHLEMRQKINDVTPCSAG',
              'DHNGPREQSMCAYTWKFLVI', 'WYLKIMFSRCTHVAQGPEND',
              'YKRTHQSPDMCLAEFVGWNI', 'WIKLFMTVCASGQNREHPDY',
              'CWIMYDLVQFHAPENRSTGK', 'PGCYNSTDIWRVHFKLQAME',
              'GPWYTNIMRFSDKHQVCALE', 'KNDEPQRSTGAHWYFLMIVC',
              'DSRQPHKWNEYVTMALFCIG', 'KDNEQSPRHTAGMFYWLIVC',
              'FMILVCWATGSPYHQNEKDR', 'MVIFLHWACRKQETYNSDGP',
              'GPYHTKSCNRVIDFWQMEAL', 'PGTSDNYCWHIFQKREVMLA',
              'PDSTGCINYEVFWRAQKHLM', 'PDNGEKSAHQRCTLMWFVYI',
              'VMILFWRAETQYCHSKDNGP', 'VIMLAYFWRCEQHKTSNDGP',
              'VLMIFYWCAHRQETKSNDGP', 'LTSAHRMWKFNYGCEVDIQP',
              'FIMLVCWHANTYGDSPERQK', 'GYNTVDHFCSIRLWQKMPEA',
              'MPAQLKSVRNEFITWDYGHC', 'LQFKIAVTDSEMRHWPYCNG',
              'WYHRKQMFENILCTDVSPAG', 'PGASCTDNVLQIEHMKFYRW',
              'PGNCYTDSHRKQEVFIWLAM', 'GSAPDNTEKQCVHIRLMYFW',
              'GPDWEYHRTCKFSANMVQIL', 'DETCVSNAGQMPLIKRHYFW',
              'KILVERDAPQGWHFMNYTCS', 'MGVNCSEYAFRLKIDQWTPH',
              'DEGKANHQFRMSLWTPCYIV', 'FWLIHVNCRYMEGDKTAQSP',
              'RHVPFIQLWTYCAKEMSNDG', 'SGYFTAPCWKIENLHQMVDR',
              'YFIWDNHVQGMERAKCLSPT', 'FILVWAMGTSYQCNPHKEDR',
              'KEQNDRYTSHPFGWILVCMA', 'RDHENKQYWSTPMCFAVILG',
              'FWILVMYPACTSQGHKREND', 'GVTAISLCMEQPDFKNHYRW',
              'RKTVQILMEYAGFSNWHPDC', 'RWTCDFESYNKMQAGHPVLI',
              'PGTCSDNHQVYIEKRFWALM', 'WCFIYVLHMATRGQSNPDEK',
              'AEMLRQKIWDVFSYHTNCGP', 'VIYFCWTLMHRSKQAEGNDP',
              'CYWVHIMFARTLPGSNDQKE', 'CWHMQFSPNYRTKDEILVAG',
              'CPWHNMQSGTYFRDVIKELA', 'KRQEPNDHSGTYAMWLIVFC',
              'QPRNESHGKDATYMILVWFC', 'PNHGSRQYAILMVKEFTWDC',
              'LERIKHMVAQFPYSWNDTCG', 'CWMHFYQITNVSALPDGRKE',
              'CWMFYIHVLPQTSNARGDKE', 'WCFMYHIVLTAPGQNSRDEK',
              'WCQHNMKDRPEYSTFGAIVL', 'WCHQKNMRDPEYSFTGIVAL',
              'CHKMEWQRPDNFTSYIGVLA', 'CWHMQNYFSTPDRIAGKVLE',
              'CMHWFRQEPYIKDLVNTSAG', 'WCHMYFNQPTRIDGSAEVKL',
              'GASCDTPNVEQHILKMRFYW', 'GASCDTPNVEQHILMKFRYW',
              'KPDESNQTGRHAYMFWLCVI', 'MDKQSNGHEPCRYTVAIWLF',
              'HYAGDPQTMLEKRSVFNWCI', 'DEGSNKPQRHTAMWCYFVIL',
              'GDPNESKQTRAHMVYCILWF', 'EKRSDQGNPHTAMWYCFLVI',
              'PSTEKQAVRDNILHFMYGCW', 'CDGNQAKSVWILTMHFREYP',
              'CGNSDWIVTAQKEHYMLFRP', 'MIWYAVGDKSHLCFTENRQP',
              'CGPTSVIDNWKHYFAQMERL', 'AWMLKCEDYINQGVSRHFTP',
              'PKESDQNRTGAMHVYLIFWC', 'GASCDTPNVEQHLIMKRFWY',
              'GASCPTDNVEQKIHLMRFWY', 'YWQCPFTIMHLNAKEVDSGR',
              'RKQNYDHEWPSTGACLFMVI', 'CIVLFMGAWSHTPYNDEQRK',
              'IVFLMACGWSTEPHYDNQKR', 'RKDEHNQSTPYWGACVILMF',
              'HKRNQSDETGAYCPMVWFLI', 'KDEPNQSRTHGAYMFWLVIC',
              'DENQGKHRSPATCWVMLIYF', 'DKQPTENSGAHWRYCMFVIL',
              'DKNPSEGTRQAWCYHMVLIF', 'KDNPSGTERAQWCHMYVLIF',
              'DKPNEQTSGARWHCYMVFIL', 'LFIMVYCHWRAGQTESPNKD',
              'LFIMVWCYAHTGRPSQNDEK', 'CILFMVWHYAGPTNDQSREK'], dtype='<U20')