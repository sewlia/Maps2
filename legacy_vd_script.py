import numpy as np
import copy


def vd(extract_time_intervals2):
    
    a = 0
    tstar = np.ones(3)
    tstar[0] = 1.1
    tstar[1]= 2.2
    tstar[2]=3.3
    Tstar = np.ones(3)
    Tstar[0] = 5.1
    Tstar[1]= 6.2
    Tstar[2]=7.3
    extract_time_intervals2_copy = copy.deepcopy(extract_time_intervals2)
    vd = np.zeros(shape=(len(extract_time_intervals2), 2))
    print('extract_time_intervals2_copy1', extract_time_intervals2_copy)
    for row_index, row in enumerate(extract_time_intervals2):
        #print('row', row)
        for col_index, (operator, start_time, end_time) in enumerate(row):
            if len(extract_time_intervals2_copy[row_index])>1:
                print(len(extract_time_intervals2_copy[row_index]))
                if operator=='G':
                    a = float(start_time)
                    vd[row_index] = a + vd[row_index] 
                    print('vd', vd)
                    print('extract_time_intervals2_copy5', extract_time_intervals2_copy)

                    extract_time_intervals2_copy[row_index].pop(0)
                    print('extract_time_intervals2_copy2', extract_time_intervals2_copy)
                    
                if operator.startswith('F'):
                    F_index = int(operator[1:]) - 1  # Extract index from operator name 'Fi'
                    a = tstar[F_index] + Tstar[F_index] 
                    vd[row_index] = a + vd[row_index]
                    print('extract_time_intervals2_copy3', extract_time_intervals2_copy)
                    print('vd1', vd)
                    extract_time_intervals2_copy[row_index].pop(0)
                    print('extract_time_intervals2_copy4', extract_time_intervals2_copy)
            else:
                if operator=='G':
                    vd[row_index] = [float(start_time), float(end_time)]
                    print('vd3', vd)
                    print('extract_time_intervals2_copy6', extract_time_intervals2_copy)
                if operator.startswith('F'):
                    F_index = int(operator[1:]) - 1  # Extract index from operator name 'Fi'
                    a = tstar[F_index] + Tstar[F_index] 
                    vd[row_index] = a + vd[row_index]
                    print('extract_time_intervals2_copy7', extract_time_intervals2_copy)
                    print('vd2', vd)             
    return vd
    if operator == 'G' and vd[row_index][0] <= midTime <= vd[row_index][1]: 
                        active = sympify(self.predicates[row_index])
                        active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                        #self.robust[row_index] = max(self.robust[row_index], active_subs)
                        self.robust[row_index] = 1000 # PlaceHolder
                        if self.robust[row_index] <= 0.05:
                            self.tau[i] = 1
                            renamed_formula_intervals_copy[row_index].pop(0)
                            
                            while len(renamed_formula_intervals_copy[row_index])>=1:
                                if operator=='G':
                                    active = sympify(self.predicates[row_index])
                                    active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                                    self.robust[row_index] = max(self.robust[row_index], active_subs)

                                    if self.robust[row_index] <= 0.05:
                                        self.tau[i] = 1
                                    renamed_formula_intervals_copy[row_index].pop(0)

                                if operator.startswith('F'):
                                    active = sympify(self.predicates[row_index])
                                    active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                                    self.robust[row_index] = min(self.robust[row_index], active_subs)
                                    if self.robust[row_index] <= 0.05:
                                        #self.tau[i] = 1
                                        self.Tstar[f_index] = midTime
                                        self.tstar[f_index] = self.Tstar[f_index]
                                        f_index += 1
                                    renamed_formula_intervals_copy[row_index].pop(0)
            
    elif operator.startswith('F') and vd[row_index][0] <= midTime <= vd[row_index][1]:
        active = sympify(self.predicates[row_index])
        active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
        self.robust[row_index] = min(self.robust[row_index], active_subs)
        print('vd[row_index][0]', vd[row_index][0])
        print('midTime', midTime)
        print('vd[row_index][1]', vd[row_index][1])
        print('f_index', f_index)
        
        if self.robust[row_index] <= 0.05:
            #self.tau[i] = 1
            self.Tstar[f_index] = midTime
            self.tstar[f_index] = self.Tstar[f_index]
            f_index += 1
            renamed_formula_intervals_copy[row_index].pop(0)
            
            while len(renamed_formula_intervals_copy[row_index])>=1:
                if operator=='G':
                    active = sympify(self.predicates[row_index])
                    active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                    self.robust[row_index] = max(self.robust[row_index], active_subs)
                    #self.robust[row_index] = 1000 # PlaceHolder
                    if self.robust[row_index] <= 0.05:
                        self.tau[i] = 1
                    renamed_formula_intervals_copy[row_index].pop(0)
                if operator.startswith('F'):
                    active = sympify(self.predicates[row_index])
                    active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                    self.robust[row_index] =  min(self.robust[row_index], active_subs)
                    if self.robust[row_index] <= 0.05:
                        self.tau[i] = 1
                        self.Tstar[f_index] = midTime
                        self.tstar[f_index] = self.Tstar[f_index]
                        f_index += 1

                    renamed_formula_intervals_copy[row_index].pop(0)

formula = [ [('G', '2', '4')], [('F1', '1.1', '2.2'), ('G', '21', '42'), ('F2', '7', '8')],[('G', '4', '6')], [('F3', '6', '8')], [('G', '8', '10')]]
print('formula', formula)
for row_index, row in enumerate(formula):
    reversed_row = row[::-1]
    print('reversed_row', reversed_row)
    for col_index, (operator, start_time, end_time) in enumerate(reversed_row):
        if len(formula[row_index])>=1:
            print('row_index', row_index)
            print('col_index', col_index)
            print('operator', operator)
            print('start_time', start_time)
            print('end_time', end_time)
            reversed_row.pop(0)
            print(reversed_row)
        else:
            print('row_index2', row_index)
            print('col_index2', col_index)
            print('operator2', operator)
            print('start_time2', start_time)
            print('end_time2', end_time)

#renamed_formula_intervals, tstar = rename_f_operators_and_init_tstar([[('F1', '1.1', '2.2'), ('G', '2', '4'), ('F2', '7', '8')], [('G', '2', '4')], [('G', '4', '6')], [('F3', '6', '8')], [('G', '8', '10')]])

def SatisfactionVariable(self, midTime, x0):
        vd = self.vd()
        #print('renamed_formula_intervals_copy', renamed_formula_intervals_copy)
        #print('vd', vd)
        #print('vd[0]', vd[0])
        #print('vd[3][0]', vd[3][0])
        #print('vd[3][1]', vd[3][1])
        #print('vd[0,1]', vd[0,1])
        #print('midTime', midTime)
        #for i, interval in enumerate(renamed_formula_intervals_copy):
        #    print('i', i)
        #    print('interval', interval)
        #    print('interval[0]', interval[0])
        #    print('interval[0][0]', interval[0][0])
        #    print('interval[0][0][0]', interval[0][0][0])
        #    if interval[0] == 'G' and vd[i][0] <= midTime <= vd[i][1]:
        #        active = sympify(self.predicates[i])
        #        active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
        #        self.robust[i] = max(self.robust[i], active_subs)
        #        #if robust[i] <= 0.05:
        #        #    tau[i] = 1
        #    elif interval[0][0][0] == 'F' and vd[i][0] <= midTime <= vd[i][1]:
        #        print('inside F')
        #        active = sympify(self.predicates[i])
        #        active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
        #        self.robust[i] = active_subs
        #        print('robust', active_subs)
                
        #        if self.robust[i] <= 0.05:
        #            self.tau[i] = 1
        #            self.Tstar = midTime
        #for row_index, row in enumerate(self.extract_time_intervals2):
        #    reversed_row = row[::-1]
        #    print('reversed_row', reversed_row)
        #    for col_index, (operator, start_time, end_time) in enumerate(reversed_row):
        #        if len(self.extract_time_intervals2[row_index])>1:
        #            if operator=='G' and self.tau[i] == -1:
        #                a = float(start_time)
        #                vd[row_index] = a + vd[row_index] 
        #                renamed_formula_intervals_copy[row_index].pop(0)
        #                i+=1
        #            if operator.startswith('F') and self.tau[i] == -1:
        #                F_index = int(operator[1:]) - 1  
        #                a = self.tstar[F_index] + self.Tstar[F_index] 
        #                vd[row_index] = a + vd[row_index]
        #                renamed_formula_intervals_copy[row_index].pop(0)
        #                i+=1
        index = 0
        f_index = 0
        renamed_formula_intervals_copy = copy.deepcopy(self.extract_time_intervals2)
        #print('renamed_formula_intervals_copy', renamed_formula_intervals_copy)
        #print('self.extract_time_intervals2', self.extract_time_intervals2)
        #print('vd', vd)
        for row_index, row in enumerate(self.extract_time_intervals2):
            reversed_row = row[::-1]
            #print('reversed_row', reversed_row)
            #print('row', row)
            for col_index, (operator, start_time, end_time) in enumerate(reversed_row):
                #print('col_index', col_index)
                i = index + len(row)  - col_index - 1
                #print('i', i)
                if len(row)==1:
                    if operator == 'G' and vd[row_index][0] <= midTime <= vd[row_index][1]: 
                        pass
                    if operator.startswith('F') and vd[row_index][0] <= midTime <= vd[row_index][1]:
                        active = sympify(self.predicates[row_index])
                        active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                        self.robust[row_index] = active_subs
                        if self.robust[row_index] <= 0.05:
                            self.Tstar[f_index] = midTime
                            self.tstar[f_index] = self.Tstar[f_index]
                            f_index += 1
                            self.tau[i] = 1
                if len(row)>1:
                    if operator == 'G' and vd[row_index][0] <= midTime <= vd[row_index][1]: 
                        pass
                    if operator.startswith('F') and vd[row_index][0] <= midTime <= vd[row_index][1]:
                        active = sympify(self.predicates[row_index])
                        active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                        self.robust[row_index] = active_subs
                        if self.robust[row_index] <= 0.05:
                            self.Tstar[f_index] = midTime
                            self.tstar[f_index] = self.Tstar[f_index]
                            f_index += 1
                            #if vd[row_index][0] != self.last_vd[row_index][0]:
                            #    self.tau[i] = -1
                            #else:
                            #    self.tau[i] = 1
                            
                    #active = sympify(self.predicates[row_index])
                    #    active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                        #self.robust[row_index] = max(self.robust[row_index], active_subs)
                    #    self.robust[row_index] = 1000 # PlaceHolder
                    #    if self.robust[row_index] <= 0.05:
                    #        self.tau[i] = 1
                    #        renamed_formula_intervals_copy[row_index].pop(0)
                            
                    #        while len(renamed_formula_intervals_copy[row_index])>=1:
                    #            if operator=='G':
                    #                active = sympify(self.predicates[row_index])
                    #                active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                    #                self.robust[row_index] = max(self.robust[row_index], active_subs)

                    #                if self.robust[row_index] <= 0.05:
                    #                    self.tau[i] = 1
                    #                renamed_formula_intervals_copy[row_index].pop(0)

                    #            if operator.startswith('F'):
                    #                active = sympify(self.predicates[row_index])
                    #                active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                    #                self.robust[row_index] = min(self.robust[row_index], active_subs)
                    #                if self.robust[row_index] <= 0.05:
                    #                    #self.tau[i] = 1
                    #                    self.Tstar[f_index] = midTime
                    #                    self.tstar[f_index] = self.Tstar[f_index]
                    #                    f_index += 1
                    #                renamed_formula_intervals_copy[row_index].pop(0)
            
                    #elif operator.startswith('F') and vd[row_index][0] <= midTime <= vd[row_index][1]:
                    #    active = sympify(self.predicates[row_index])
                    #    active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                    #    self.robust[row_index] = min(self.robust[row_index], active_subs)
                    #    print('vd[row_index][0]', vd[row_index][0])
                    #    print('midTime', midTime)
                    #    print('vd[row_index][1]', vd[row_index][1])
                    #    print('f_index', f_index)
                        
                    #    if self.robust[row_index] <= 0.05:
                    #        #self.tau[i] = 1
                    #        self.Tstar[f_index] = midTime
                    #        self.tstar[f_index] = self.Tstar[f_index]
                    #        f_index += 1
                    #        renamed_formula_intervals_copy[row_index].pop(0)
                    #        
                    #        while len(renamed_formula_intervals_copy[row_index])>=1:
                    #            if operator=='G':
                    #                active = sympify(self.predicates[row_index])
                    #                active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                    #                self.robust[row_index] = max(self.robust[row_index], active_subs)
                    #                #self.robust[row_index] = 1000 # PlaceHolder
                    #                if self.robust[row_index] <= 0.05:
                    #                    self.tau[i] = 1
                    #                renamed_formula_intervals_copy[row_index].pop(0)
                    #            if operator.startswith('F'):
                    #                active = sympify(self.predicates[row_index])
                    #                active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                    #                self.robust[row_index] =  min(self.robust[row_index], active_subs)
                    #                if self.robust[row_index] <= 0.05:
                    #                    self.tau[i] = 1
                    #                    self.Tstar[f_index] = midTime
                    #                    self.tstar[f_index] = self.Tstar[f_index]
                    #                    f_index += 1#

                    #                renamed_formula_intervals_copy[row_index].pop(0)
            index = index + len(row)
        self.last_vd = vd