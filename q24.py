import math

import numpy as np


def q24():
    with open("2023-24.txt") as f:
        lines = f.read().splitlines()

    hailstones = []
    for line in lines:
        position_raw, velocity_raw = line.split(' @ ')
        position = np.array([int(n) for n in position_raw.split(', ')])
        velocity = np.array([int(n) for n in velocity_raw.split(', ')])
        hailstones.append({
            'p': position,
            'v': velocity
        })

    # Part 1

    boundary_min = 200000000000000
    boundary_max = 400000000000000

    def inverse(matrix):
        try:
            return np.linalg.inv(matrix)
        # Handle on our own since we expect this to happen
        except np.linalg.LinAlgError:
            return None

    total_crosses = 0
    for i1 in range(len(hailstones)):
        h1 = hailstones[i1]
        p1 = h1['p']
        v1 = h1['v']
        for i2 in range(i1 + 1, len(hailstones)):
            h2 = hailstones[i2]
            p2 = h2['p']
            v2 = h2['v']

            # We start from v1 * t1 + p1 = v2 * t2 + p2
            # We can then rewrite it as:
            #     | v1_0, -v2_0 |   | t1 |   | p2_0 - p1_0 |
            #     | v1_1, -v2_1 | x | t2 | = | p2_1 - p1_1 |
            # An equation of the form A * x = B has solution x = A^(-1) * B

            A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
            A_inverse = inverse(A)

            # Both lines have the same slope and thus will never cross
            if A_inverse is None:
                continue

            B = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            product = np.matmul(A_inverse, B)

            if (product <= 0).any():
                # The crossing happened in the past for at least one line
                continue

            t1 = product[0]
            x, y, _ = p1 + v1 * t1

            is_outside_boundary = False
            for variable in (x, y):
                if variable < boundary_min or variable > boundary_max:
                    # The crossing happened outside the boundary
                    is_outside_boundary = True
                    break
            if is_outside_boundary:
                continue

            total_crosses += 1

    # Part 2
            
    # - Let v_alpha and p_alpha be vectors that satisfy the following:
    #       For every hailstone h_i with velocity v_i and initial position p_i, there exists
    #       a time t_i > 0 s.t. t_i v_alpha + p_alpha = t_i * v_i + p_i
    # - We can rewrite each case as v_alpha + (1 / t_i) * p_alpha = v_i + (1 / t_i) * p_i
    # - Let chi_i (Greek letter chosen arbitrarily) = p_alpha x p_i (cross product)
    # - Then, (v_alpha + (1 / t_i) * p_alpha) • chi_i = (v_i + (1 / t_i) * p_i) • chi_i
    # - When we distribute the dot product and get rid of zero terms, we get:
    #       v_alpha • (p_alpha x p_i) = v_i • (p_alpha x p_i)
    # - Using the triple product property to rewrite both sides yields:
    #       p_i • (v_alpha x p_alpha) = p_alpha • (p_i x v_i)
    # - We have the information to calculate p_i x v_i
    # - For simplicity, let's write v_alpha x p_alpha as vp_alpha and p_i x v_i as pv_i
    # - Now, let's bring in p_j • vp_alpha = p_alpha • pv_j for j =/= i
    # - When we multiply equations by scalars s_i and s_j and subtract them, we get:
    #       (s_i * p_i - s_j * p_j) • vp_alpha = p_alpha • (s_i * pv_i - s_j * pv_j)
    # - If we choose s_i and s_j carefully and repeat this, we can cancel out the vp_alpha on the left
    #   and end up with equations of the form:
    #       0 = p_alpha • (some constant vector)
    # - If we get at least two such constant vectors and take the cross product, we'll get p_alpha,
    #   or at least a vector that is a scalar multiple of p_alpha
            
    # TODO: Cleanup?

    # No Numpy! Numpy cuts off numbers...

    # Writing all these in-house whoo!
    def cross_product(v1, v2):
        assert len(v1) == 3
        assert len(v2) == 3
        v_i = v1[1] * v2[2] - v1[2] * v2[1]
        v_j = v1[2] * v2[0] - v1[0] * v2[2]
        v_k = v1[0] * v2[1] - v1[1] * v2[0]
        return [v_i, v_j, v_k]
    
    def scalar_multiply(s, v):
        return [s * v_i for v_i in v]
    
    def vector_subtract(v1, v2):
        assert len(v1) == len(v2)
        to_return = []
        for i in range(len(v1)):
            to_return.append(v1[i] - v2[i])
        return to_return
    
    def scalar_vector_subtract(s1, v1, s2, v2):
        return vector_subtract(
            scalar_multiply(s1, v1),
            scalar_multiply(s2, v2)
        )
    
    def reduce_vector(v):
        if not v:
            return v

        gcd = v[0]
        for i in range(1, len(v)):
            gcd = math.gcd(gcd, v[i])

        if not gcd:
            return v

        reduced = [vi // gcd for vi in v]

        first_nonzero_i = 0
        while first_nonzero_i < len(v) and v[first_nonzero_i] == 0:
            first_nonzero_i += 1
        # Can't be len(v), because if the whole array was zeros gcd would return 0 and short-circuit above
        if reduced[first_nonzero_i] < 0:
            reduced = scalar_multiply(-1, reduced)

        return reduced
    
    def unique_vectors(v_list):
        if not len(v_list):
            return []
        intermediate = sorted(v_list)
        to_return = [intermediate[0]]
        for i in range(1, len(intermediate)):
            if intermediate[i] != intermediate[i - 1]:
                to_return.append(intermediate[i])
        return to_return
    
    hailstones = []
    for line in lines:
        position_raw, velocity_raw = line.split(' @ ')
        position = [int(n) for n in position_raw.split(', ')]
        velocity = [int(n) for n in velocity_raw.split(', ')]
        hailstones.append({
            'p': position,
            'v': velocity,
            'p x v': cross_product(position, velocity)
        })

    # We don't actually have to go through all the points to calculate the starting position vector!
    n = 6


    vector_equality_pairs = []
    for i1 in range(n):
        h1 = hailstones[i1]
        p1 = h1['p']
        cp1 = h1['p x v']
        for i2 in range(i1 + 1, n):
            h2 = hailstones[i2]
            p2 = h2['p']
            cp2 = h2['p x v']
            first = scalar_vector_subtract(p2[0], p1, p1[0], p2)
            second = scalar_vector_subtract(p2[0], cp1, p1[0], cp2)
            vector_equality_pairs.append((first, second))

    for i in range(1, 3):
        intermediate = []
        for i1 in range(n):
            x1 = vector_equality_pairs[i1][0]
            y1 = vector_equality_pairs[i1][1]
            for i2 in range(i1 + 1, n):
                x2 = vector_equality_pairs[i2][0]
                y2 = vector_equality_pairs[i2][1]
                first = scalar_vector_subtract(x2[i], x1, x1[i], x2)
                second = scalar_vector_subtract(x2[i], y1, x1[i], y2)
                intermediate.append((first, second))
        vector_equality_pairs = intermediate

    final = []
    for x, y in vector_equality_pairs:
        assert x == [0, 0, 0]
        if y != [0, 0, 0]:
            final.append(reduce_vector(y))
    final = unique_vectors(final)

    choices = []
    for i in range(len(final)):
        for j in range(i + 1, len(final)):
            choices.append(reduce_vector(cross_product(final[i], final[j])))
    choices = unique_vectors(choices)
    # When done correctly, by construction of the problem there's a unique line and thus position
    # that hits all the others
    assert len(choices) == 1
    choice = choices[0]

    # We calculate the position vector as: [239756157786030, 463222539161932, 273997500449219]
    # With similar code we calculate the slope vector as: [47, -360, 18]
    # The code below verifies that these vectors are indeed a solution
    p_alpha = [239756157786030, 463222539161932, 273997500449219]
    v_alpha = [47, -360, 18]
    for h in hailstones:
        p_difference = vector_subtract(
            p_alpha,
            h['p']
        )
        v_difference = vector_subtract(
            h['v'],
            v_alpha
        )
        t = p_difference[0] // v_difference[0]
        assert t > 0 and (scalar_multiply(t, v_difference) == p_difference)

    return total_crosses, sum(choice)