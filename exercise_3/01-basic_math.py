from decimal import Decimal, getcontext
import sys
import math
import random

class Polynomial:
    def __init__(self, coefficients):
        """
        Initialize a polynomial.

        coefficients: list of coefficients in ascending order,
                      i.e. [a0, a1, a2, ...] represents a0 + a1*x + a2*x^2 + ...
        """
        self.coefficients = coefficients
        self._trim()
    
    def _trim(self):
        # Remove trailing zeros to update the polynomial's degree.
        while len(self.coefficients) > 1 and abs(self.coefficients[-1]) < 1e-12:
            self.coefficients.pop()
    
    def degree(self):
        return len(self.coefficients) - 1
    
    def __call__(self, x):
        # Evaluate the polynomial at x using Horner's method.
        result = 0
        for coef in reversed(self.coefficients):
            result = result * x + coef
        return result
    
    def derivative(self):
        # Return a new Polynomial that is the derivative of this polynomial.
        if self.degree() == 0:
            return Polynomial([0])
        new_coeffs = [i * self.coefficients[i] for i in range(1, len(self.coefficients))]
        return Polynomial(new_coeffs)
    
    def solve(self, precision=20):
        """
        Solve the polynomial exactly when its degree is less than 4.
        
        For degree 3, initial roots are estimated via Cardano's method;
        then each root is refined using Newton's method with Decimal (arbitrary precision).
        
        Returns:
            - For a constant polynomial: "All real numbers are solutions" if 0, otherwise None.
            - For a linear polynomial: the unique solution.
            - For a quadratic polynomial: a tuple of solution(s).
            - For a cubic polynomial: a tuple with refined roots.
            - Otherwise: None.
        """
        deg = self.degree()
        if deg == 0:
            if abs(self.coefficients[0]) <= 0.0:
                return "All real numbers are solutions"
            else:
                return None
        elif deg == 1:
            a0, a1 = self.coefficients[0], self.coefficients[1]
            if abs(a1) <= 0.0:
                return None
            return -a0 / a1
        elif deg == 2:
            a0, a1, a2 = self.coefficients[0], self.coefficients[1], self.coefficients[2]
            if abs(a2) < 1e-12:
                # Degenerates to linear.
                if abs(a1) < 1e-12:
                    return None
                return -a0 / a1
            discriminant = a1**2 - 4 * a2 * a0
            if discriminant < 0:
                # Return complex roots.
                real_part = -a1 / (2 * a2)
                imag_part = math.sqrt(-discriminant) / (2 * a2)
                return (complex(real_part, imag_part), complex(real_part, -imag_part))
            else:
                root1 = (-a1 + math.sqrt(discriminant)) / (2 * a2)
                root2 = (-a1 - math.sqrt(discriminant)) / (2 * a2)
                if abs(discriminant) < 1e-12:
                    return root1
                return (root1, root2)
        elif deg == 3:
            # Cubic case: use Cardano's formula.
            # Write cubic: a*x^3 + b*x^2 + c*x + d = 0.
            a, b, c, d = self.coefficients[3], self.coefficients[2], self.coefficients[1], self.coefficients[0]
            if abs(a) < 1e-12:
                return None  # Should have been caught as a quadratic.
            # Change variable: x = t - b/(3a)
            shift = b / (3 * a)
            # Depressed cubic: t^3 + p*t + q = 0,
            p = (3 * a * c - b**2) / (3 * a**2)
            q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
            # Calculate discriminant.
            delta = (q / 2)**2 + (p / 3)**3

            def cube_root(x):
                if x >= 0:
                    return x**(1/3)
                else:
                    return -(-x)**(1/3)

            # Compute initial t-roots based on delta.
            if abs(delta) < 1e-12:
                t1 = 2 * cube_root(-q / 2)
                t2 = -cube_root(-q / 2)
                initial_roots = [t1 - shift, t2 - shift]
            elif delta > 0:
                t = cube_root(-q / 2 + math.sqrt(delta)) + cube_root(-q / 2 - math.sqrt(delta))
                real_root = t - shift
                omega = complex(-0.5, math.sqrt(3) / 2)
                t2 = cube_root(-q / 2 + math.sqrt(delta)) * omega + cube_root(-q / 2 - math.sqrt(delta)) * omega.conjugate()
                t3 = cube_root(-q / 2 + math.sqrt(delta)) * omega.conjugate() + cube_root(-q / 2 - math.sqrt(delta)) * omega
                initial_roots = [real_root, t2 - shift, t3 - shift]
            else:
                # Three distinct real roots.
                r_val = math.sqrt(-(p / 3)**3)
                theta = math.acos(-q / (2 * math.sqrt(-(p / 3)**3)))
                t1 = 2 * math.sqrt(-p / 3) * math.cos(theta / 3)
                t2 = 2 * math.sqrt(-p / 3) * math.cos((theta + 2 * math.pi) / 3)
                t3 = 2 * math.sqrt(-p / 3) * math.cos((theta + 4 * math.pi) / 3)
                initial_roots = [t1 - shift, t2 - shift, t3 - shift]

            getcontext().prec = precision
            # Define Newton's method for refinement using Decimal for arbitrary precision.
            def newton_opt_dec(x0, tol=Decimal('1e-20'), max_iter=100):
                # Evaluate the polynomial in Decimal using Horner's method.
                def poly_dec(x):
                    x = Decimal(x)
                    res = Decimal(0)
                    for coef in reversed(self.coefficients):
                        res = res * x + Decimal(coef)
                    return res
                # Evaluate the derivative in Decimal.
                def poly_deriv_dec(x):
                    if self.degree() == 0:
                        return Decimal(0)
                    der_coeffs = [Decimal(i) * Decimal(self.coefficients[i]) for i in range(1, len(self.coefficients))]
                    x = Decimal(x)
                    res = Decimal(0)
                    for coef in reversed(der_coeffs):
                        res = res * x + coef
                    return res
                
                x = Decimal(x0)
                for _ in range(max_iter):
                    fx = poly_dec(x)
                    dfx = poly_deriv_dec(x)
                    if abs(dfx) < tol:
                        break
                    new_x = x - fx / dfx
                    if abs(new_x - x) < tol:
                        return new_x
                    x = new_x
                return x

            # Refine each initial root using the Decimal-based Newton refinement.
            # Only refine if all initial roots are real numbers
            if all(isinstance(r, (int, float)) and not isinstance(r, complex) for r in initial_roots):
                refined_roots = sorted([newton_opt_dec(r) for r in initial_roots])
            else:
                refined_roots = initial_roots
            return tuple(refined_roots)
        else:
            return None

    def __repr__(self):
        # Generates a human-readable representation of the polynomial.
        terms = []
        for i, coef in enumerate(self.coefficients):
            if abs(coef) < 1e-12:
                continue
            if i == 0:
                terms.append(f"{coef}")
            elif i == 1:
                terms.append(f"{coef}*x")
            else:
                terms.append(f"{coef}*x^{i}")
        if not terms:
            return "0"
        return " + ".join(terms)

# read in lines from standard input
num_lines = int(sys.stdin.readline().strip())
# num_lines = 10
for _ in range(num_lines):
    """
    # testing
    p = random.uniform(0, 1)
    if p < 0.7:
        uppper_bound = 5e6
        x = random.randint(1, int(uppper_bound))
        y = random.randint(1, int(uppper_bound))
        z = y   # test multiple roots
        # z = random.randint(1, int(uppper_bound))
        print(f"Solutions: {' '.join(map(str, sorted([x, y, z])))}")
        U = x+y+z
        V = x*y*z
        W = x**2 + y**2 + z**2
    else:
        uppper_bound = 5e18
        U = random.randint(1, int(uppper_bound))
        V = random.randint(1, int(uppper_bound))
        W = random.randint(1, int(uppper_bound))
    """
    U, V, W = map(int, sys.stdin.readline().strip().split())
    
    # create polynomial x^3 - U*x^2 + (U^2-W)/2*x^2 - V
    coefficients = [-V, (U**2 - W) // 2, -U, 1]
    p = Polynomial(coefficients)
    # print(f"Polynomial: {p}")
    # solve the polynomial
    roots = p.solve(precision=18)
    #print(f"Roots: {roots}")
    # Check if all roots are distinct positive integers
    if roots is None or isinstance(roots, str):
        print("empty_set")
    else:
        if not isinstance(roots, (tuple, list)):
            roots = (roots,)
        int_roots = []
        for r in roots:
            # If r is a Decimal, check directly.
            if isinstance(r, Decimal):
                rounded = r.to_integral_value()
                int_roots.append(int(rounded))
            else:
                break
        if len(set(int_roots)) == 3:
            result = sorted(int_roots)
            print(" ".join(map(str, result)))
        else:
            print("empty set")