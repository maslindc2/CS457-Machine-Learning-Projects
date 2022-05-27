""" Genetic Algorithm created for maximizing a given function
    Maslin Farrell, Isabelle Melton, Nick Fechtel"
"""
from numpy.random import rand, randint

def geneticAlgorithm(function, bounds, numBits, numIterations, numPopulation,  crossoverRate, mutationRate):
    # Population of random bitstrings
    population = [randint(0,2, numBits * len(bounds)).tolist() for _ in range(numPopulation)]
    
    # Results list used for plotting the results of f(x,y) values for each generation
    results = list() # Used for all generations and f(x,y) values
    resultsBest = list() # Used for storing the new best f(x,y) value

    # Store best solutions
    best, bestEval = 0, function(decode(bounds, numBits, population[0]))
    
    # Enumerate generations
    for gen in range(numIterations):
        #Decode the population
        decoded = [decode(bounds, numBits, p) for p in population]

        # Evaluate all candidates in the population
        scores = [function(d) for d in decoded]
        for i in range(numPopulation):
            if scores[i] > bestEval:
                best, bestEval = population[i], scores[i]
                print("Gen: %d, new maximum f(%s) = %f" % (gen, decoded[i], scores[i]))
                resultsBest.append([gen, scores[i]])

        # select parents
        selectedParents = [tournamentSelection(population, scores) for _ in range(numPopulation)]
        
        # Append current generation and current score for plotting 
        results.append([gen, scores[gen]])
        
        # Create next generation
        children = list()
        for i in range(0, numPopulation, 2):
            
            # Get the selected parents in pairs
            parent1, parent2 = selectedParents[i], selectedParents[i+1]

            # crossover and mutation
            for c in crossover(parent1, parent2, crossoverRate):
                mutation(c, mutationRate)

                # store child for next generation
                children.append(c)
        # Replace population
        population = children

    return [results, resultsBest, best, bestEval]

# Decode the bitstring to numbers so we can understand what the x and y values are
def decode(bounds, numBits, bitstring):
    decoded = list()
    largest = 2**numBits
    for i in range(len(bounds)):
        # Extract the substring
        start, end = i * numBits, (i * numBits) + numBits
        substring = bitstring[start:end]
        # Convert the bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # Convert string to integer
        integer = int(chars, 2)
        # Scale integer to desired range
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        # Store the decoded bitstring 
        decoded.append(value)
    return decoded

# Tournament selection function
def tournamentSelection(population, scores, k=3):
    #first random selection
    selection = randint(len(population))
    for i in randint(0, len(population), k-1):
        #perform a tournament if the current score is better
        if scores[i] > scores[selection]:
            selection = i
    return population[selection]

# Crossover two parents to create two children
def crossover(parent1, parent2, crossoverRate):
    #Children are copies of their parents
    child1, child2 = parent1.copy(), parent2.copy()
    if rand() < crossoverRate:
        # pick crossover point
        cPoint = randint(1, len(parent1)-2)
        # Do crossover
        child1 = parent1[:cPoint] + parent2[cPoint:]
        child2 = parent2[:cPoint] + parent1[cPoint:]
    return [child1, child2]

# Mutation function
def mutation(bitstring, mutationRate):
    for i in range(len(bitstring)):
        # Check for a mutation
        if rand() < mutationRate:
            #flip the current bit
            bitstring[i] = 1 - bitstring[i]

#Used for getting the x and y values used for plotting the evolution of f(x,y) values
def fetchPlotValues(resultList):
    x = list()
    y = list()

    # Split the result list into two lists for x values and y values
    for i in range(len(resultList)):
        if(i % 2) == 0: # Even indexes are X values
            x.append(resultList[i])
        else: # Odd indexes are Y values
            y.append(resultList[i]) 
    return x, y # return it as one assembled list