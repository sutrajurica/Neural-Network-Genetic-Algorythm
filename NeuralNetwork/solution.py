import numpy as np# importamo potrebne libraryje
import sys
import csv
import random

def get_err(y_true, y_predicted): # racunamo srednje kvadratno odstupanje
    mse = np.mean((y_true - y_predicted) ** 2)
    # y_true = test_data[:, -1:]
    return mse

def sigmoida(x): # funkcija logističke sigmoide
    return 1 / (1 + np.exp(-x))

def generate_Matrix_W_and_b(input_dim, output_dim):
    dim = (input_dim, output_dim) # dimenzija matrice tezina

    weights = np.random.normal(loc=0.0, scale=0.01, size=dim) # uzorkujte iz normalne razdiobe sa standardnom devijacijom 0.01

    dim = (1, output_dim)

    biases = np.random.normal(loc=0.0, scale=0.01, size=dim)

    return weights, biases

def network_to_chromosome(neural_network): # iz mreže pravimo kromosom
    flat_parts = []
    
    for key in sorted(neural_network.keys()):
        W = neural_network[key][0]  # uzmi matricu tezina
        b = neural_network[key][1]  # uzmi matricu biasa
        
        flat_parts.append(W.flatten())  # spljosti tezine i dodaj u listu
        flat_parts.append(b.flatten())  # spljosti biase i dodaj u listu
        
    # spoji sve u kromosom
    return np.concatenate(flat_parts)

def chromosome_to_network(sample_network, chromosome): 
    reconstructed_network = {}
    
    #brojac pamti dokle smo stigli u kromosomu
    idx = 0 

    for key in sorted(sample_network.keys()):
        W, b = sample_network[key]
        
        # uzmi broj elemenata za tezinu W
        W_num = W.size
        # Režemo od trenutnog položaja do (trenutni + koliko nam treba)
        chromosome_W = chromosome[idx : idx + W_num]
        matrix_W = chromosome_W.reshape(W.shape) #shape je tuple s dimenzijama
        idx += W_num
        
        b_num = b.size
        chromosome_b = chromosome[idx : idx + b_num]
        matrix_b = chromosome_b.reshape(b.shape)
        idx += b_num

        reconstructed_network[key] = [matrix_W, matrix_b]
        
    return reconstructed_network

def mutate_chromosome(chromosome, p, K):
    maska = np.random.rand(chromosome.size) < p 
    #za svako mjesto u kromosomu provjeri je li nasumicni broj < p
    # np.random.rand(chromosome.size) vraca brojeve od 0.0 do 1.0
    # ako je size npr 15, pravi vektor duljine 15 u kojem ce biti true ili false na svakom mjestu
    # otprilike p posto elemenata je true

    gauss = np.random.normal(loc=0.0, scale=K, size=chromosome.size)

    chromosome[maska] += gauss[maska]

    return chromosome

def roulette_wheel(chromosome_list):
    fitness_sum = 0

    for i in chromosome_list: # zbrajamo sve fitnesse
        fitness, chromosome = i
        fitness_sum += fitness

    parent1 = None # init roditelje
    parent2 = None

    spin1 = random.random() # biramo broj izmedju 0 i 1
    spin2 = random.random()

    track_parent1 = 0
    track_parent2 = 0

    for i in chromosome_list:
        fitness, chromosome = i
        length = fitness/fitness_sum

        track_parent1 += length

        if track_parent1 > spin1:
            parent1 = chromosome
            break

    for i in chromosome_list:
        fitness, chromosome = i
        length = fitness/fitness_sum

        track_parent2 += length

        if track_parent2 > spin2:
            parent2 = chromosome
            break

    if parent1 is None:
        parent1 = chromosome_list[-1][1]
    if parent2 is None:
        parent2 = chromosome_list[-1][1]

    return parent1, parent2
    

class Genetic_Algorythm:
    def __init__(self, train_data, test_data, nn, popsize, elitism, p, K, iteracija, header): # postavimo sve vrijednosti za konstruktor
        self.train_data = train_data
        self.test_data = test_data
        self.nn = nn
        self.popsize = int(popsize)
        self.elitism = int(elitism)
        self.p = float(p)
        self.K = float(K)
        self.iteracija = int(iteracija)
        self.header = header
        self.neural_network = None

    def fit(self):
        self.neural_network = self.generate_neural_network(self.nn, self.header)
        self.neural_network = self.genetic_alg(self.nn, self.header, self.popsize)
    
    def genetic_alg(self, nn, header, popsize):
        chromosome_list = []

        for i in range(popsize): # stvaramo pocetnu populaciju, sortirana od najveceg fitnessa do najmanjeg
            neural_n = self.generate_neural_network(nn, header)
            y_predicted = self.predict(self.train_data[:, :-1], neural_n)

            err = get_err(self.train_data[:,-1:], y_predicted)
            fitness = 1/err

            chromosome = network_to_chromosome(neural_n)

            chromosome_touple = (fitness, chromosome)

            chromosome_list.append(chromosome_touple)

        chromosome_list = sorted(chromosome_list, key=lambda x: x[0], reverse=True)

        counter = 0

        while (counter != self.iteracija + 1):
            new_generation = []

            for i in range(self.elitism): # najbolja (ili viˇse najboljih) jedinki se prenosi u iducu generaciju
                new_generation.append(chromosome_list[i])

            while(len(new_generation) != self.popsize):
                chromosome1_touple, chromosome2_touple = roulette_wheel(chromosome_list) # odabiremo roditelje iz originalne populacije

                parent1 = chromosome1_touple
                parent2 = chromosome2_touple

                child = (parent1 + parent2) / 2.0 # krizanje
                child = mutate_chromosome(child, self.p, self.K) # mutacija

                child = self.chromosome_touple(child)

                new_generation.append(child)

            chromosome_list = new_generation
            chromosome_list = sorted(chromosome_list, key=lambda x: x[0], reverse=True)

            if (counter % 2000 == 0 and counter != 0): #svakih 2000 generacija ispisujemo srednje kvadratno odstupanje najbolje jedinke na skupu podataka za treniranje
                best_touple = chromosome_list[0]
                fitness, chromosome = best_touple

                err = fitness ** -1

                print(f"[Train error @{counter}] : {err}")

            counter += 1

        min_err = float('inf')

        for i in range(popsize):
            chromosome_touple = chromosome_list[i] # na kraju ispisujemo srednje kvadratno odstupanje najbolje jedinke na skupu za testiranje
            fitness, chromosome = chromosome_touple

            neural_n = chromosome_to_network(self.neural_network, chromosome)
            y_predicted = self.predict(self.test_data[:, :-1], neural_n)
            err = get_err(self.test_data[:,-1:], y_predicted)

            if err < min_err:
                min_err = err
        
        print(f"[Test error] : {min_err}")

        best_fitness, best_chromosome = chromosome_list[0]
        best_network = chromosome_to_network(self.neural_network, best_chromosome)

        return best_network 

    def chromosome_touple(self, chromosome): # pravimo tople (fitness, chromosome)
        new_network = chromosome_to_network(self.neural_network, chromosome)
        y_predicted = self.predict(self.train_data[:, :-1], new_network)
        err = get_err(self.train_data[:,-1:], y_predicted)
        fitness = 1/err
        chromosome = network_to_chromosome(new_network)
        chromosome_touple_ret = (fitness, chromosome)

        return chromosome_touple_ret

    def generate_neural_network(self, nn, header): # napravimo rijecnik {0 : [W1, b1]}, klucevi su redni brojevi sloja
        arch = nn.split("s")
        arch1 = []

        for layer in arch: # izoliramo koliko u svakom skrivenom sloju ima neurona
            if layer != '':
                layer = int(layer)
                arch1.append(layer)

        entry_layer_num = [len(header) - 1] # gledamo koliko imamo ulaza
        arch1 = entry_layer_num + arch1
        arch1.append(1) # konacno imamo listu koja nam govori koliko u svakom sloju imamo neurona
        arch = arch1

        neural_network_dict = {} # gradimo neuronsku mrezu

        layer_num = 0
        for i in range (len(arch) - 1):
            input_dim = arch[i]
            output_dim = arch[i + 1]
            weights, biases = generate_Matrix_W_and_b(input_dim, output_dim)
            value_list = []
            value_list.append(weights)
            value_list.append(biases)
            neural_network_dict[layer_num] = value_list
            layer_num += 1

        return neural_network_dict
             
    def predict(self, X, weights_dict=None):
        if weights_dict is None: # ako pozivamo izvan funckije fit
            weights_dict = self.neural_network
        
        trenutna_X = X
        num_layers = len(weights_dict)
    
        for i in range(num_layers): # provedi kroz neuronsku mrezu, napiši čovjeku na papir
            W, b = weights_dict[i]
            net = np.dot(trenutna_X, W) + b
        
            if i < num_layers - 1:
                trenutna_X = sigmoida(net)
            else:
                trenutna_X = net
            
        return trenutna_X
    
def main():
    args = sys.argv[1:]
    train = None # argumenti predani komandnom linijom
    test = None
    nn = None
    popsize = None
    elitism = None
    p = None
    K = None
    iteracija = None

    for i in range (len(args)): # parsiramo argumente komandne linije
        if args[i] == "--train":
            train = args[i + 1]
        if args[i] == "--test":
            test = args[i + 1]
        if args[i] == "--nn":
            nn = args[i + 1]
        if args[i] == "--popsize":
            popsize = args[i + 1]
        if args[i] == "--elitism":
            elitism = args[i + 1]
        if args[i] == "--p":
            p = args[i + 1]
        if args[i] == "--K":
            K = args[i + 1]
        if args[i] == "--iter":
            iteracija = args[i + 1]

    with open(train, mode='r', encoding='utf-8') as f: # ucitavamo datoteku train
        reader = csv.reader(f)
        header = next(reader)
        train_data = np.array(list(reader), dtype=float)

    with open(test, mode='r', encoding='utf-8') as f: # ucitavamo datoteku test
        reader = csv.reader(f)
        _header = next(reader)
        test_data = np.array(list(reader), dtype=float)
    
    model = Genetic_Algorythm(train_data, test_data, nn, popsize, elitism, p, K, iteracija, header) # pozivamo klasu genetskog algoritma
    model.fit() # pravimo neuronsku mrezu
    #predicted = model.predict(train_data[:, :-1])
    #print(predicted)
if __name__ == "__main__":
    main()