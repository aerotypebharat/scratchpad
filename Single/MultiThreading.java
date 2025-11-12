package com.god.file.individual;

import java.util.concurrent.Semaphore;

public class MultiThreading {

    // Pattern 31: MULTI-THREAD
    // Use Case: Concurrency, synchronization


    // 31.1: Print in Order

    class Foo {
        private Semaphore firstDone = new Semaphore(0);
        private Semaphore secondDone = new Semaphore(0);

        public Foo() {
        }

        public void first(Runnable printFirst) throws InterruptedException {
            printFirst.run();
            firstDone.release();
        }

        public void second(Runnable printSecond) throws InterruptedException {
            firstDone.acquire();
            printSecond.run();
            secondDone.release();
        }

        public void third(Runnable printThird) throws InterruptedException {
            secondDone.acquire();
            printThird.run();
        }
    }


    // 31.2: Print FooBar Alternately

    class FooBar {
        private int n;
        private Semaphore fooSem = new Semaphore(1);
        private Semaphore barSem = new Semaphore(0);

        public FooBar(int n) {
            this.n = n;
        }

        public void foo(Runnable printFoo) throws InterruptedException {
            for (int i = 0; i < n; i++) {
                fooSem.acquire();
                printFoo.run();
                barSem.release();
            }
        }

        public void bar(Runnable printBar) throws InterruptedException {
            for (int i = 0; i < n; i++) {
                barSem.acquire();
                printBar.run();
                fooSem.release();
            }
        }
    }


    // 31.3: Dining Philosophers

    class DiningPhilosophers {
        private Semaphore[] forks = new Semaphore[5];
        private Semaphore dining = new Semaphore(4); // Allow only 4 philosophers to eat

        public DiningPhilosophers() {
            for (int i = 0; i < 5; i++) {
                forks[i] = new Semaphore(1);
            }
        }

        public void wantsToEat(int philosopher, Runnable pickLeftFork, Runnable pickRightFork, Runnable eat, Runnable putLeftFork, Runnable putRightFork) throws InterruptedException {

            int leftFork = philosopher;
            int rightFork = (philosopher + 1) % 5;

            dining.acquire();

            forks[leftFork].acquire();
            forks[rightFork].acquire();

            pickLeftFork.run();
            pickRightFork.run();

            eat.run();

            putLeftFork.run();
            putRightFork.run();

            forks[leftFork].release();
            forks[rightFork].release();

            dining.release();
        }
    }

}
