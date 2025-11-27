import java.util.*;
import java.util.stream.*;

class Employee {
    String name;
    String department;
    double salary;
    int age;
    int id;
    
    Employee(String name, String department, double salary, int age) {
        this.name = name;
        this.department = department;
        this.salary = salary;
        this.age = age;
    }
    
    Employee(String name, int id, double salary, long joinDate) {
        this.name = name;
        this.id = id;
        this.salary = salary;
    }
}

class Person {
    String name;
    int age;
    String city;
    
    Person(String name, int age, String city) {
        this.name = name;
        this.age = age;
        this.city = city;
    }
    
    Person(Person other) {
        this.name = other.name;
        this.age = other.age;
        this.city = other.city;
    }
}

public class JavaStreamPractice {
    
    // 1. RANGE OPERATIONS
    public static void rangeExamples() {
        IntStream.rangeClosed(1, 10).forEach(n -> System.out.print(n + " "));
        System.out.println();
        
        int sum = IntStream.rangeClosed(1, 100).sum();
        System.out.println("Sum 1-100: " + sum);
        
        IntStream.rangeClosed(1, 20)
                 .filter(n -> n % 2 == 0)
                 .forEach(n -> System.out.print(n + " "));
        System.out.println();
    }
    
    // 2. MAPTO OPERATIONS
    public static void mapToExamples() {
        List<String> numberStrings = IntStream.rangeClosed(1, 5)
                .mapToObj(n -> "Number-" + n)
                .collect(Collectors.toList());
        System.out.println("Number strings: " + numberStrings);
        
        List<Person> people = IntStream.rangeClosed(20, 25)
                .mapToObj(age -> new Person("Person-" + age, age, "City" + age))
                .collect(Collectors.toList());
        System.out.println("People created: " + people.size());
        
        List<String> strings = Arrays.asList("apple", "banana", "cherry");
        int totalLength = strings.stream().mapToInt(String::length).sum();
        System.out.println("Total length: " + totalLength);
        
        int[] lengths = strings.stream().mapToInt(String::length).toArray();
        System.out.println("Lengths array: " + Arrays.toString(lengths));
    }
    
    // 3. GROUPINGBY OPERATIONS
    public static void groupingByExamples() {
        List<String> words = Arrays.asList("apple", "banana", "apricot", "berry");
        
        Map<Character, List<String>> byFirstLetter = words.stream()
                .collect(Collectors.groupingBy(word -> word.charAt(0)));
        System.out.println("Grouped by first letter: " + byFirstLetter);
        
        Map<Integer, Long> countByLength = words.stream()
                .collect(Collectors.groupingBy(
                    String::length, 
                    Collectors.counting()
                ));
        System.out.println("Count by length: " + countByLength);
    }
    
    // 4. PARTITIONINGBY OPERATIONS
    public static void partitioningByExamples() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        Map<Boolean, List<Integer>> evenOdd = numbers.stream()
                .collect(Collectors.partitioningBy(n -> n % 2 == 0));
        System.out.println("Even/Odd partition: " + evenOdd);
        
        Map<Boolean, Long> countByLength = Arrays.asList("apple", "banana", "cat", "dog").stream()
                .collect(Collectors.partitioningBy(
                    word -> word.length() > 4,
                    Collectors.counting()
                ));
        System.out.println("Word length partition count: " + countByLength);
    }
    
    // 5. MAPTO PRIMITIVE ARRAYS
    public static void mapToPrimitiveExamples() {
        List<String> numberStrings = Arrays.asList("1", "2", "3", "4", "5");
        int[] intArray = numberStrings.stream().mapToInt(Integer::parseInt).toArray();
        System.out.println("String to int[]: " + Arrays.toString(intArray));
        
        List<String> bigNumbers = Arrays.asList("10000000000", "20000000000");
        long[] longArray = bigNumbers.stream().mapToLong(Long::parseLong).toArray();
        System.out.println("String to long[]: " + Arrays.toString(longArray));
        
        List<String> decimals = Arrays.asList("1.5", "2.7", "3.14");
        double[] doubleArray = decimals.stream().mapToDouble(Double::parseDouble).toArray();
        System.out.println("String to double[]: " + Arrays.toString(doubleArray));
        
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", 101, 75000.50, 0L),
            new Employee("Bob", 102, 65000.75, 0L)
        );
        
        int[] employeeIds = employees.stream().mapToInt(emp -> emp.id).toArray();
        System.out.println("Employee IDs: " + Arrays.toString(employeeIds));
    }
    
    // 6. COMPREHENSIVE EXAMPLE
    public static void comprehensiveExample() {
        List<Employee> employees = Arrays.asList(
            new Employee("Alice", "IT", 75000, 28),
            new Employee("Bob", "HR", 55000, 32),
            new Employee("Charlie", "IT", 80000, 35)
        );
        
        Map<String, Double> avgSalaryByDept = employees.stream()
                .collect(Collectors.groupingBy(
                    e -> e.department,
                    Collectors.averagingDouble(e -> e.salary)
                ));
        System.out.println("Avg salary by dept: " + avgSalaryByDept);
        
        Map<Boolean, List<Employee>> byAge = employees.stream()
                .collect(Collectors.partitioningBy(e -> e.age >= 30));
        System.out.println("Employees by age partition: " + byAge.size() + " groups");
        
        DoubleSummaryStatistics stats = employees.stream()
                .mapToDouble(e -> e.salary)
                .summaryStatistics();
        System.out.printf("Salary stats: count=%d, avg=%.2f\n", stats.getCount(), stats.getAverage());
    }
    
    // 7. REST/SPREAD PATTERNS
    public static void restSpreadExamples() {
        System.out.println("Sum with varargs: " + sum(1, 2, 3, 4, 5));
        
        List<Integer> combined = spreadLists(Arrays.asList(1, 2, 3), Arrays.asList(4, 5, 6));
        System.out.println("Combined lists: " + combined);
    }
    
    public static int sum(int... numbers) {
        return Arrays.stream(numbers).sum();
    }
    
    public static <T> List<T> spreadLists(List<T> list1, List<T> list2) {
        return Stream.concat(list1.stream(), list2.stream()).collect(Collectors.toList());
    }

    public static void main(String[] args) {
        System.out.println("=== Java Stream Practice ===\n");
        
        System.out.println("1. Range Operations:");
        rangeExamples();
        
        System.out.println("\n2. MapTo Operations:");
        mapToExamples();
        
        System.out.println("\n3. GroupingBy Operations:");
        groupingByExamples();
        
        System.out.println("\n4. PartitioningBy Operations:");
        partitioningByExamples();
        
        System.out.println("\n5. MapTo Primitive Arrays:");
        mapToPrimitiveExamples();
        
        System.out.println("\n6. Comprehensive Employee Analysis:");
        comprehensiveExample();
        
        System.out.println("\n7. Rest/Spread Patterns:");
        restSpreadExamples();
        
        System.out.println("\n=== Practice Complete ===");
    }
}