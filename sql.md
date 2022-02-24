# SQL and Databases

### Some general pointers on SQL

- Order of occurence: select, where, group by, having
- Commands related to transaction control in SQL: Rollback, Commit, Savepoint
- In a relational schema, there exists only one primary key and it can't take any null values. 
- Difference between primary key and unique key: In a relational schema, you can have only one primary key. But there can be multiple unique keys
- Difference between Truncate, Delete and Drop: Truncate and Drop are DDL statements- faster,cannot be rolled back, and the storage space is released. While Delete command is a DML command, can be rolled back and space is not deallocated.
- In DBMS, records are also known as tuple/rows. Columns on the other hand are called attributes/fields
- IS statement is used to compare NULL 
- Difference between Having and Group By- Having is performed after group by. 
- ANY and ALL operate on subqueries. ANY returns true if any of the subquery values meet the condition. ALL returns true if all the conditions are true.
- A relation satisfying xNF will automatically satisfy yNF, if x < y. Eg 2NF will satisfy the conditions of 1NF
- Difference between Super, Candidate and Primary Key: The minimal set of attributes that can uniquely identify a tuple is known as the candidate key. The set of attributes that can identify a tuple is a Super key. Super key is a superset of Candidate key. There can be more than one candidate key in a relationship, out of which one can be choosen as the primary key.
- Like is case sensitive whereas, iLike is not
- ACID properties in DBMS:
    - Atomicity: The entire transaction takes place at once or doesn't happen at all. This prevents any partial update which can be problematic.
    - Consistency: The database must be consistent before and after the transaction. 
    - Isolation: Multiple transactions occur independently without interference.
    - Durability: The changes of a successful transaction occurs even if the system failure occurs.
- CTEs and subqueries- They allow to subset the data and store that with a name- which can be subsequently used for other select commands
- 'select count(colx)' is less than or equal to 'select count(*)': The first case escapes the NULL values in column 'colx'
- 'sum' treats nulls as 0
- SQL evaluates the aggregations before the 'limit' clause
- Use 'Having' clause to filter on aggregate columns
- Using 'case' clause with aggegations- Use this when you want to aggregate rows that fulfil a certain condition
- You can also use Case statements inside the aggregation functions
    - Example:
    ```sql
    select 
    count(case when year = 'FR' then 1 else NULL END) as fr_counts,
    count(case when year = 'SO' then 1 else NULL END) as so_counts
    from table_x;

    ```
- A conditional statement like 'And' can be used along with a Join statement. It will be evaluated before the join occurs. Filtering in the where clause does almost the same- except it filters the result in both the tables after join is performed
- Union clause removes the duplicates from both the tables whereas Union All appends all the values in both the tables
- SQL joins using where or On produces different results- because in the first case the join happens first and then the rows are filtered, whereas in the second case the rows are joined that fit the 'On' condition
- Couple of reasons why you might want to join tables based on multiple keys:
    - One being accuracy
    - Second is performance. SQL uses indexes to speed up queries
- Difference between DDL, DML, DCL and TCL
    - DDL: Data Definition Language- Create, Drop, Alter, Truncate
    - DML: Data Manipulation Language- Select, Insert, Update, Delete
    - DCL: Data Control Language- Grant, Evoke
    - TCL: Transaction Control Language- Commands applieb by database admins- to ensure data consistency and avoid loss

### Helpful Data Manipulation functions

- LEFT(string, number of characters)
- TRIM(both '()' FROM location): Removes the characters within the quotes from the location
- POSITION('A' IN descript) or STRPOS(descript, 'A'): Returns char position from left
- SUBSTR(*string*, *starting character position*, *# of characters*): Returns the substring within the string defined
- CONCAT(day_of_week, ', ', LEFT(date, 10) OR day_of_week || ', ' || LEFT(date, 10): String concatenation
- EXTRACT('year'   FROM cleaned_date) AS year : Extracts the part from date formated field. Can be year, month, day, hour, second, dow
- DATE_TRUNC('year'   , cleaned_date): Rounds to any precision (Eg- Year, month, day etc)
- COALESCE(descript,'No Description'): Bypass NULL values

### Helpful Window functions

- Running total
    - Example:
    ```sql
    SUM(duration_seconds) OVER (ORDER BY start_time) AS running_total
    ```
- Running total within partition
    - Example:
    ```sql
    SUM(duration_seconds) OVER(Partition by start_terminal order by start_time) as running_total_st
    ```
- Getting row number
    - Example:
    ```sql
    ROW_NUMBER() OVER (Partition by start_terminal order by start_time) as row_num_st
    ```
- Rank gives indentical rank in case of a tie, while dense rank will give rank to the identical rows, but no rank is skipped
    - Example:
    ```sql
    RANK() OVER (Partition by start_terminal order by start_time) as rank
    DENSE_RANK() OVER (Partition by start_terminal order by start_time) as d_rank

    ```
- Convert to different buckets:
    - NTILE(4)- quartiles
    - NTILE(100)- percentiles
    - NTILE(buckets) OVER (partition by x order by y)
- Lag/Lead
    - LAG(duration_seconds, 1) - Pulls previous records
    - LEAD(duration_seconds, 1) - Pulls next records
    - calculate differences between rows
        - Example:
        ```sql
        duration_seconds -LAG(duration_seconds, 1) OVER
            (PARTITION BY start_terminal ORDER BY duration_seconds)
            AS difference
        ```

- Conditional where clause example:
```sql
WHERE
  (TRUNC(DATE_VAR2) - TRUNC(DATE_VAR1)) >=   CASE 
                                                WHEN SEQ_VAR IN (1,2,3) THEN 0 
                                                WHEN SEQ_VAR IN (4,5,6) THEN 1 ELSE 2 
                                                END 

```

### Some pointers on Optimizing queries

- Ways to improve performance:
    - Reduce table size by filtering the data that you need
    - Make joins less complicated   
    - Adding EXPLAIN at the start of the query generates a query plan 
- Optimizing query: hen you want to use query hints, restructure the query, update statistics, use temporary tables, add indexes, and so on to get better performance.
- CTE vs Subqueries vs Temp Tables: temp is materalized and CTE is not.CTE is just syntax so in theory it is just a subquery. It is executed. #temp is materialized. So an expensive CTE in a join that is execute many times may be better in a #temp. On the other side if it is an easy evaluation that is not executed but a few times then not worth the overhead of #temp. CTE for readability and performance. Performance may not kick in until the CTE is used twice, but if the second JOIN has to be built in, your syntax will allow for that development more easily.

## Examples and Solutions

### Write a SQL query to get the second highest salary from the Employee table.

Table: Employee

| id | salary |
|----|--------|
| 1  | 3000   |
| 2  | 2000   |
| 3  | 5000   |

Solution:

```sql
SELECT 
max(salary) 
from Employee
where salary != (SELECT 
                max(salary) 
                from Employee
                );
```

### Write a SQL query to find all duplicate emails in a table named Employee.

Table: Employee

| id | email    |
|----|----------|
| 1  | a@xy.com |
| 2  | b@xy.com |
| 3  | a@xy.com |

Solution:

```sql
select email 
from (
    select 
    email,
    count(*) as email_count
    from Employee
    group by email
)
where email_count>1;
OR
select 
email
from Employee 
group by email
having count(email)>1;
```

### Write a sql query to find all the dates where the prices are higher than its previous day.

Table: stock_price

| id | date       | price |
|----|------------|-------|
| 1  | 2020-01-01 | 134   |
| 2  | 2020-01-02 | 136   |
| 3  | 2020-01-03 | 132   |
| 4  | 2020-01-04 | 130   |

Solution:

```sql
with t1 as 
(
select
date,
price as todays_price,
lag (price, 1) over(order by date asc) as yesterdays_price
from stock_price
),
select 
id,date
from t1
where todays_price>yesterdays_price
;
--OR
select
a.id,
a.date
from stock_price a join stock_price b -- a = today, b= yesterday
on a.id = b.id
where 1=1
and a.price>b.price
datediff(a.date,b.date)=1
;
```

### Write a query to find the highest salary in each of the departments.

Table: Employee

| id | salary | department |
|----|--------|------------|
| 1  | 3000   | A          |
| 2  | 2000   | B          |
| 3  | 5000   | B          |
| 4  | 4000   | A          |

Solution:

```sql
with t1 as (
select
id,
salary,
rank() over (partition by department order by salary desc) as rank_salary
from employee
)
select 
id,salary
from t1
where rank_salary =1
;
```

### Exchange alternate positions for the table Seating.

Table: Seating

| id | name    |
|----|----------|
| 1  | Adam     |
| 2  | John     |
| 3  | Mary     |
| 4  | James    |
| 5  | Andy     |

Expected Output:

| id | name    |
|----|----------|
| 1  | John     |
| 2  | Adam     |
| 3  | James    |
| 4  | Mary     |
| 5  | Andy     |

Solution:

- If total rows are odd, then it remains unchanged
- In rest of the cases, 
    - the odd id is assigned its next id
    - the even id is assigned its previous id

```sql
select 
case 
    when ((select max(id)%2 from seating) = 1 and  id = (select max(seat) from seating) then id
    when id%2 =1 then id+1
    else id-1
end as id,
name
from seating
order by id;
case
``` 

### Write a query to get departments top three salaries. ([Leetcode link](https://leetcode.com/problems/department-top-three-salaries/))

Input: Employee table


| id | name  | salary | departmentId |
|----|-------|--------|--------------|
| 1  | Joe   | 85000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
| 5  | Janet | 69000  | 1            |
| 6  | Randy | 85000  | 1            |
| 7  | Will  | 70000  | 1            |


Department table:


| id | name  |
|----|-------|
| 1  | IT    |
| 2  | Sales |

Output: 

| Department | Employee | Salary |
|------------|----------|--------|
| IT         | Max      | 90000  |
| IT         | Joe      | 85000  |
| IT         | Randy    | 85000  |
| IT         | Will     | 70000  |
| Sales      | Henry    | 80000  |
| Sales      | Sam      | 60000  |

Solution:

```sql
with t1 as 

(
select 
e.Id,
e.name,
e.Salary,
e.departmentid,
d.name as dept_name
from Employee e  
join department d on e.departmentid = d.id

),t2 as (
select
t1.id,
t1.name as 'Employee',
t1.dept_name,
t1.Salary,
DENSE_RANK() OVER(partition by dept_name order by salary desc) as rank_dept
from t1

)
select 
dept_name as 'Department',
Employee,
Salary 
from t2 where 
rank_dept <=3
order by rank_dept
```

### Pivot the Occupation column in OCCUPATIONS so that each Name is sorted alphabetically and displayed underneath its corresponding Occupation. The output column headers should be Doctor, Professor, Singer, and Actor, respectively.(Ref[Hackerrank Link ](https://www.hackerrank.com/challenges/occupations/problem))

Table: Occupation 

| name     | occupation |
|----------|------------|
| Samantha | Doctor     |
| Julia    | Actor      |
| Maria    | Actor      |
| Katy     | Professor  |

Solution:

```sql
SELECT d.Name, p.Name, s.Name, a.Name 
FROM 
(SELECT Name, Row_number() over(order by name) as num FROM Occupations WHERE Occupation = 'Doctor') d 
full join 
(SELECT Name, Row_number() over(order by name) as num FROM Occupations WHERE Occupation = 'Professor') p 
on d.num = p.num 
full join 
(SELECT Name, Row_number() over(order by name) as num FROM Occupations WHERE Occupation = 'Singer') s 
on d.num = s.num or p.num = s.num 
full join 
(SELECT Name, Row_number() over(order by name) as num FROM Occupations WHERE Occupation = 'Actor') a 
on d.num = a.num or p.num = a.num or s.num = a.num
```

to be continued...




